import itertools
import os
import time
import warnings

import numpy as np
import torch
import torchvision
from torchvision.io import VideoReader, read_video_timestamps, read_video
from torch.utils.data import DataLoader, Dataset
from torchaudio import transforms as audio_T
from torchvision import transforms as frame_T
from torchvision.transforms import _transforms_video as transforms_video
from utils.transforms import *

from torchvision.models.video import R3D_18_Weights
from torchaudio.pipelines import WAV2VEC2_BASE, WAV2VEC2_LARGE, WAV2VEC2_LARGE_LV60K

from CrossModalGraph.configs.config import get_cfg

from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as GraphDataLoader


def get_transforms(cfg):
    # video_frame_transform = frame_T.Compose(
    #     [
    #         frame_T.Resize((112, 112)),
    #         # frame_T.Normalize(
    #         #     mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]
    #         # ),
    #     ]
    # )
    video_frame_transform = None
    audio_transform = torch.nn.Sequential(
        audio_T.Resample(44100, WAV2VEC2_BASE.sample_rate),
        # audio_T.MelSpectrogram(sample_rate=16000, n_mels=128),
    )
    video_transform = frame_T.Compose(
        [
            video_frame_resample(
                new_fps=20,
            ),
            # Resize(112),
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], channel=-1),
        ]
    )
    return video_frame_transform, audio_transform, video_transform


class AudioSetGraphDataset(Dataset):
    """AudioSet graph dataset."""

    def __init__(
        self,
        root: str,
        video_frame_transform=None,
        audio_transform=None,
        video_transform=None,
        graph_transform=None,
        clip_len=10,
        desired_classes=["All"],
        config=None,
        verbose=False,
        **kwargs,
    ):
        self.root = root
        self.video_frame_transform = video_frame_transform
        self.audio_transform = audio_transform
        self.video_transform = video_transform
        self.graph_transform = graph_transform
        self.clip_len = clip_len
        self.verbose = verbose

        # Graph memory
        self.Graph_memory = []

        self.video_path, self.labels = self._get_video_files(root)

        # pick unique labels
        self.unique_labels = list(set(self.labels))
        self.unique_labels.sort()

        # filter data based on the classes
        if desired_classes != ["All"]:
            self.desired_idx = [
                idx for idx, label in enumerate(self.labels) if label in desired_classes
            ]
            self.video_path = [self.video_path[idx] for idx in self.desired_idx]
            self.labels = [self.labels[idx] for idx in self.desired_idx]
            self.label_dict = self.label2idx = {
                label: idx for idx, label in enumerate(desired_classes)
            }
            self.idx2label = {idx: label for idx, label in enumerate(desired_classes)}
        else:
            self.label_dict = self.label2idx = {
                label: idx for idx, label in enumerate(self.unique_labels)
            }
            self.idx2label = {
                idx: label for idx, label in enumerate(self.unique_labels)
            }

        # create numerical labels
        self.numerical_labels = [self.label2idx[label] for label in self.labels]

        weights = R3D_18_Weights.DEFAULT
        if "video_frame_resample" in self.video_transform.transforms[0].__repr__():
            self.video_transform.transforms.append(weights.transforms())

        self._init_graph_param_(graph_config=config.GRAPH)

    def _get_video_files(self, root):
        # pick up all the video files in sub-folders in the root directory
        video_files = []
        labels = []
        for folder in os.listdir(root):
            for file in os.listdir(os.path.join(root, folder)):
                if file.endswith(".mp4"):
                    video_files.append(os.path.join(root, folder, file))
                    labels.append(folder)
        return video_files, labels

    def __len__(self):
        return len(self.video_path)

    def _init_graph_param_(self, graph_config):
        self.num_vid_nodes = graph_config.NUM_VIDEO_NODES
        self.num_aud_nodes = graph_config.NUM_AUDIO_NODES
        assert (
            graph_config.SPAN_OVER_TIME_AUDIO > 0
        ), "span_over_time_audio must be greater than 0"
        assert (
            graph_config.SPAN_OVER_TIME_VIDEO > 0
        ), "span_over_time_video must be greater than 0"
        assert (
            graph_config.SPAN_OVER_TIME_BETWEEN > 0
        ), "span_over_time_between must be greater than 0"
        self.span_over_time_audio = (
            graph_config.SPAN_OVER_TIME_AUDIO
        )  # np.random.randint(low=2, high=graph_config.SPAN_OVER_TIME_AUDIO + 1)
        self.span_over_time_video = (
            graph_config.SPAN_OVER_TIME_VIDEO
        )  # np.random.randint(low=2, high=graph_config.SPAN_OVER_TIME_VIDEO + 1)
        self.span_over_time_between = graph_config.SPAN_OVER_TIME_BETWEEN
        self.audio_dilation = graph_config.AUDIO_DILATION
        self.video_dilation = graph_config.VIDEO_DILATION
        self.dynamic = graph_config.DYNAMIC
        self.normalize = graph_config.NORMALIZE  # make attributes normalized
        self.self_loops = graph_config.SELF_LOOPS  # add self loops

        # np.random.seed(seed)
        self.hop_length = np.random.randint(
            np.ceil(self.num_aud_nodes / self.num_vid_nodes)
        )
        # compute the residual audio nodes
        self.resd = max(
            int(self.num_aud_nodes - (self.num_vid_nodes - 1) * self.hop_length) - 2, 0
        )
        # set a residual list for considering extra hop
        self.hop_prob_list = np.random.randint(
            np.floor(self.resd / self.num_vid_nodes),
            np.ceil(self.resd / self.num_vid_nodes) + 1,
            self.num_vid_nodes,
        )
        # make sure that the first video node connects to the first audio node
        self.hop_prob_list[0] = 0
        # make sure that the last video node will reach the end of the audio
        if self.resd - self.hop_prob_list.sum() > 0:
            self.hop_prob_list_comp = np.array(
                [1] * (self.resd - self.hop_prob_list.sum())
                + [0] * (self.num_vid_nodes - (self.resd - self.hop_prob_list.sum()))
            )
            # shuffle the list
            np.random.shuffle(self.hop_prob_list_comp)

            self.hop_prob_list = self.hop_prob_list + self.hop_prob_list_comp

    def _create_init_graph_(self):
        if not self.dynamic and len(self.Graph_memory) != 0:
            return self.Graph_memory
        else:
            # set the span over time for each modality
            span_over_time_audio = (
                np.random.randint(low=1, high=self.span_over_time_audio + 1)
                if self.dynamic
                else self.span_over_time_audio
            )
            span_over_time_video = (
                np.random.randint(low=1, high=self.span_over_time_video + 1)
                if self.dynamic
                else self.span_over_time_video
            )
            span_over_time_between = (
                np.random.randint(low=1, high=self.span_over_time_between + 1)
                if self.dynamic
                else self.span_over_time_between
            )
            video_dilation = (
                np.random.randint(low=1, high=self.video_dilation + 1)
                if self.dynamic
                else self.video_dilation
            )
            audio_dilation = (
                np.random.randint(low=1, high=self.audio_dilation + 1)
                if self.dynamic
                else self.audio_dilation
            )

            graph = HeteroData()

            # creating edges
            vid_sr_edges = []
            vid_dt_edges = []
            aud_sr_edges = []
            aud_dt_edges = []
            aud_vid_sr_edges = []
            aud_vid_dt_edges = []

            # creating video edges
            for i in range(self.num_vid_nodes):
                start_idx = i - span_over_time_video * video_dilation
                end_idx = 1 + i + span_over_time_video * video_dilation
                dt_edges = list(range(start_idx, end_idx, video_dilation))

                # remove negative index
                while dt_edges[0] < 0:
                    dt_edges.pop(0)
                    # make sure all nodes have same number of neighbors (same degree)
                    dt_edges.append(dt_edges[-1] + video_dilation)
                # remove the last indices if they are out of bound
                while dt_edges[-1] >= self.num_vid_nodes:
                    dt_edges.pop(-1)
                    # make sure all nodes have same number of neighbors (same degree)
                    dt_edges.insert(0, dt_edges[0] - video_dilation)
                # remove self loops if not self_loops
                if not self.self_loops and i in dt_edges:
                    dt_edges.remove(i)

                sr_edges = [i] * len(dt_edges)
                vid_dt_edges += dt_edges
                vid_sr_edges += sr_edges
            vid_edges = np.array([vid_dt_edges, vid_sr_edges])

            # creating audio edges
            for i in range(self.num_aud_nodes):
                start_idx = i - span_over_time_audio * audio_dilation
                end_idx = 1 + i + span_over_time_audio * audio_dilation
                dt_edges = list(range(start_idx, end_idx, audio_dilation))

                # remove negative index
                while dt_edges[0] < 0:
                    dt_edges.pop(0)
                    dt_edges.append(dt_edges[-1] + audio_dilation)
                while dt_edges[-1] >= self.num_aud_nodes:
                    dt_edges.pop(-1)
                    # make sure all nodes have same number of neighbors (same degree)
                    dt_edges.insert(0, dt_edges[0] - audio_dilation)
                # remove self loops if not self_loops
                if not self.self_loops:
                    dt_edges.remove(i)

                sr_edges = [i] * len(dt_edges)
                aud_dt_edges += dt_edges
                aud_sr_edges += sr_edges
            aud_edges = np.array([aud_dt_edges, aud_sr_edges])

            # keep track of hop shift
            hop_shift = 0
            for i in range(self.num_vid_nodes):
                hop_shift = min(hop_shift + self.hop_prob_list[i], self.resd)
                start_idx = min(
                    int(hop_shift + i * self.hop_length), self.num_aud_nodes - 1
                )
                end_idx = min(
                    int(start_idx + span_over_time_between), self.num_aud_nodes
                )
                dt_edges = list(range(start_idx, end_idx))
                # if not self.self_loops: dt_edges.remove(i)
                sr_edges = [i] * len(dt_edges)
                aud_vid_dt_edges += dt_edges
                aud_vid_sr_edges += sr_edges
            aud_vid_edges = np.array([aud_vid_sr_edges, aud_vid_dt_edges])

            graph["video", "video-video", "video"].edge_index = torch.from_numpy(
                vid_edges
            )
            graph["audio", "audio-audio", "audio"].edge_index = torch.from_numpy(
                aud_edges
            )
            # the task is acoustics classification so we need to add the edges from video to audio
            graph["video", "video-audio", "audio"].edge_index = torch.from_numpy(
                aud_vid_edges
            )
            graph["video"].x = torch.zeros(self.num_vid_nodes, 1)
            graph["audio"].x = torch.zeros(self.num_aud_nodes, 1)

            sample = {
                "graph": graph,
                "span_over_time_between": span_over_time_between,
                "span_over_time_audio": span_over_time_audio,
                "audio_dilation": audio_dilation,
                "span_over_time_video": span_over_time_video,
                "video_dilation": video_dilation,
                "num_vid_nodes": self.num_vid_nodes,
                "num_aud_nodes": self.num_aud_nodes,
            }

            if self.graph_transform:
                sample = self.graph_transform(sample)

            if not self.dynamic:
                self.Graph_memory = sample

            return sample

    def __getitem__(
        self,
        idx,
    ):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        audio_fps = None
        video_fps = None

        video_address = self.video_path[idx]
        video, audio, info = read_video(
            filename=video_address,
            pts_unit="sec",
            start_pts=0,
            end_pts=None,
            output_format="TCHW",
        )

        T, C, H, W = video.shape

        # if video fps less than 10, skip
        if info["video_fps"] < 20:
            upsampled_video = video.permute(1, 2, 3, 0)  # C, H, W, T
            upsampled_video = upsampled_video.reshape(C, H * W, T)
            scale_factor = int(np.ceil(20 / info["video_fps"]))
            upsampler = torch.nn.Upsample(
                scale_factor=scale_factor, mode="nearest-exact"
            )
            upsampled_video = upsampler(upsampled_video.float())
            upsampled_video = upsampled_video.reshape(
                C, H, W, -1
            )  # C, H, W, scale_factor*T
            video = upsampled_video.permute(3, 0, 1, 2)  # scale_factor*T, C, H, W
            video = video.to(torch.uint8)
            info["video_fps"] = info["video_fps"] * scale_factor

        video_len = int(video.shape[0] / info["video_fps"])  # in seconds

        # if read video length is less than the clip length, then pad the video
        if (
            video.shape[0] < int(self.clip_len * info["video_fps"])
            or video_len < self.clip_len
        ):
            if self.verbose:
                warnings.warn(
                    "The video is shorter than the clip length, the video will be padded with zeros"
                )
            padding_len = int(self.clip_len * info["video_fps"]) - video.shape[0]
            video = torch.cat(
                (
                    video,
                    torch.zeros(
                        padding_len,
                        video.shape[1],
                        video.shape[2],
                        video.shape[3],
                    ),
                ),
                0,
            )
            video_len = self.clip_len

        audio_len = int(audio.shape[1] / info["audio_fps"])  # in seconds

        # if read audio length is less than the clip length, then pad the audio with zeros
        if (
            audio.shape[1] < int(self.clip_len * info["audio_fps"])
            or audio_len < self.clip_len
        ):
            if self.verbose:
                warnings.warn(
                    "The audio is shorter than the clip length, the audio will be padded with zeros"
                )
            padding_len = int(self.clip_len * info["audio_fps"]) - audio.shape[1]
            audio = torch.cat(
                (
                    audio,
                    torch.zeros(
                        audio.shape[0],
                        padding_len,
                    ),
                ),
                1,
            )
            audio_len = self.clip_len

        # cut audio if it is longer than the video length
        if audio.shape[1] > int(audio_len * info["audio_fps"]):
            audio = audio[:, : int(video_len * info["audio_fps"])]

        # apply video transforms
        if self.video_transform:
            if "video_frame_resample" in self.video_transform.transforms[0].__repr__():
                self.video_transform.transforms[0].original_fps = info["video_fps"]
            video = self.video_transform(video)  # C, T, H, W
            if "video_frame_resample" in self.video_transform.transforms[0].__repr__():
                video_fps = self.video_transform.transforms[0].new_fps
            else:
                video_fps = video.shape[1] / video_len
        else:
            video_fps = info["video_fps"]

        # cut video frames if it is longer than the clip length
        if video.shape[1] > int(self.clip_len * video_fps):
            video = video[:, : int(self.clip_len * video_fps), :, :]
        elif video.shape[1] < int(self.clip_len * video_fps):
            if self.verbose:
                warnings.warn(
                    "The video is shorter than the clip length, the video will be padded with zeros"
                )
            padding_len = int(self.clip_len * video_fps) - video.shape[1]
            video = torch.cat(
                (
                    video,
                    torch.zeros(
                        video.shape[0],
                        padding_len,
                        video.shape[2],
                        video.shape[3],
                    ),
                ),
                1,
            )
        if video.shape[1] != 200:
            a = 0

        # apply audio transforms
        if self.audio_transform:
            if self.audio_transform._modules["0"]._get_name() == "Resample":
                self.audio_transform._modules["0"].orig_freq = info["audio_fps"]
            audio = self.audio_transform(audio)
            audio_fps = np.floor(audio.shape[1] / video_len)
        else:
            audio_fps = info["audio_fps"]

        if audio.shape[1] != 160000:
            a = 0

        # apply frame transforms
        C, T, H, W = video.shape
        if self.video_frame_transform:
            new_frames = []
            for i in range(video.shape[1]):
                new_frames.append(
                    self.video_frame_transform(video[:, i, :, :].float()).unsqueeze(0)
                )
                # # to print and validate the transforms
                # import matplotlib.pyplot as plt
                # plt.imshow(video[i, :, :, :].numpy())
                # plt.figure()
                # plt.imshow(new_frames[0][0, :, :, :].permute(1, 2, 0).to(torch.uint8).numpy())
            video = torch.cat(new_frames, 0)

        sample = {
            "video_path": video_address,
            "video": video,
            "audio": audio,
            "video_fps": video_fps,
            "audio_fps": audio_fps,
            "audio_len": audio_len,
            "video_len": video_len,
            "numerical_label": self.label_dict[self.labels[idx]],
        }

        # create initial heterogeneous graph
        graph_dict = self._create_init_graph_()

        # update sample dict with graph dict
        sample.update(graph_dict)

        return sample


if __name__ == "__main__":

    cfg = get_cfg()
    cfg.merge_from_file("../configs/AudioSet.yaml")

    # create transforms
    video_frame_transform, audio_transform, video_transform = get_transforms(cfg)

    # create dataset
    dataset = AudioSetGraphDataset(
        root="/media/amir_shirian/Amir/Datasets/Sound_Recognition/AudioSet/new_Eval_full_audioset",
        video_frame_transform=video_frame_transform,
        audio_transform=audio_transform,
        video_transform=video_transform,
        # desired_classes=["Accelerating, revving, vroom"],
        config=cfg,
    )
    dataloader = GraphDataLoader(dataset, batch_size=100, shuffle=True, num_workers=0)
    # measure taken time
    start = [time.time()]
    for i_batch, sample_batched in enumerate(dataloader):
        print(
            "Batch ",
            i_batch,
            sample_batched["video"].size(),
            sample_batched["audio"].size(),
        )
        start.append(time.time())

    print(
        "Time taken: ",
        np.array(start[0]) - np.array(start[-1]),
        "average: ",
        np.mean(np.array(start[1:]) - np.array(start[:-1])),
    )
