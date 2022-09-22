from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from kornia.losses import focal_loss
from scipy import signal
from scipy.signal import savgol_filter
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn
from torch.nn.init import xavier_uniform_
from torch_geometric.data import HeteroData
# from CrossModalGraph.model.GNN_models import GATConv
from torch_geometric.nn import (GATConv, GATv2Conv, GCNConv, GlobalAttention,
                                GraphNorm, InstanceNorm, LayerNorm, Linear,
                                PairNorm, SAGEConv, TransformerConv,
                                global_max_pool, global_mean_pool)
from torchaudio.pipelines import (WAV2VEC2_BASE, WAV2VEC2_LARGE,
                                  WAV2VEC2_LARGE_LV60K)
from torchvision.models.video import R3D_18_Weights, r3d_18

from CrossModalGraph.configs.config import configurable
from CrossModalGraph.model.build import META_ARCH_REGISTRY
# from torch_geometric.nn import HeteroConv
from CrossModalGraph.model.utils import (HeteroConv, num_of_graphs,
                                         pairwise_distances)
from CrossModalGraph.structures.instances import Instances
from CrossModalGraph.utils.events import get_event_storage
from CrossModalGraph.utils.utils import atten_dive_score

# from torch_geometric.nn.glob.gmt import GraphMultisetTransformer



__all__ = ["EndToEndHeteroGNN"]


@META_ARCH_REGISTRY.register()
class EndToEndHeteroGNN(nn.Module):
    """
    End to end model for multimodal graph cross modal learning.
    """

    @configurable
    def __init__(
        self,
        hidden_channels: int = 128,
        out_dim: int = 128,
        num_layers: int = 2,
        device: str = "cpu",
        normalize: bool = False,
        self_loops: bool = False,
        vis_period: int = 0,
        loss: str = "CrossEntropyLoss",
        cfg: Optional[Dict] = None,
    ):
        """
        Args:
            hidden_channels (int): number of hidden channels
            out_dim (int): output dimension (number of classes)
            num_layers (int): number of heterogenous layers
            device (str): device to use
            normalize (bool): whether to normalize the node features
            self_loops (bool): whether to add self-loops to the graph
            vis_period: the period to run visualization. Set to 0 to disable.
            loss (str): loss function to use
        """
        super().__init__()

        self.vis_period = vis_period
        self.loss = loss
        self.device = device
        self.out_dim = out_dim
        self.fusion_layers = np.array(cfg.GRAPH.FUSION_LAYERS) + num_layers
        self.residual = cfg.GRAPH.RESIDUAL

        self.num_aud_nodes = cfg.GRAPH.NUM_AUDIO_NODES
        self.num_vid_nodes = cfg.GRAPH.NUM_VIDEO_NODES

        self.audio_seg_len = cfg.GRAPH.AUDIO_SEGMENT_LENGTH
        self.video_seg_len = cfg.GRAPH.VIDEO_SEGMENT_LENGTH

        # video head
        weights = R3D_18_Weights.DEFAULT
        self.video_head = r3d_18(weights=weights).to(device)
        # self.video_head.avgpool.register_forward_hook(get_features('feats'))
        if cfg.MODEL.VIDEO_BACKBONE.FINETUNE:
            for param in self.video_head.parameters():
                param.requires_grad = True
            self.video_head.train()
        else:
            for param in self.video_head.parameters():
                param.requires_grad = False
            self.video_head.eval()

        # adio head; refer here for more options (https://pytorch.org/audio/stable/pipelines.html)
        # self.audio_head = WAV2VEC2_BASE.get_model().to(
        #     device
        # )  # usage: self.audio_head.extract_features(x)
        self.audio_head = WAV2VEC2_BASE.get_model().feature_extractor.to(
            device
        )  # usage: self.audio_head.extract_features(x)
        if cfg.MODEL.AUDIO_BACKBONE.FINETUNE:
            for param in self.audio_head.parameters():
                param.requires_grad = True
            self.audio_head.train()
        else:
            for param in self.audio_head.parameters():
                param.requires_grad = False
            self.audio_head.eval()

        # define conv layer for each edge
        self.convs = torch.nn.ModuleList()
        self.Norm_layers = []
        for i in range(num_layers):

            # check the layer type
            if i in self.fusion_layers:
                audio_conv = video_conv = GCNConv(
                    -1,
                    hidden_channels,
                    normalize=normalize,
                    add_self_loops=self_loops,
                )
                # audio_conv = video_conv = TransformerConv(-1, hidden_channels)
            else:
                audio_conv = GCNConv(
                    -1,
                    hidden_channels,
                    normalize=normalize,
                    add_self_loops=self_loops,
                )
                video_conv = GCNConv(
                    -1,
                    hidden_channels,
                    normalize=normalize,
                    add_self_loops=self_loops,
                )

            # define a heterogeneous conv layer
            conv = HeteroConv(
                {
                    # ('video', 'video-video', 'video'): GCNConv(-1, hidden_channels),
                    # ('audio', 'audio-audio', 'audio'): GATConv((-1, -1), hidden_channels),
                    # ('audio', 'audio-video', 'video'): SAGEConv((-1, -1), hidden_channels),
                    ("video", "video-video", "video"): video_conv,
                    # ('video', 'video-video', 'video'): TransformerConv(-1, hidden_channels),
                    ("audio", "audio-audio", "audio"): audio_conv,
                    # the task is acoustics classification so we need to add the edges from video to audio
                    ("video", "video-audio", "audio"): GATConv(
                        (-1, -1),
                        hidden_channels,
                        add_self_loops=self_loops,
                        heads=1,
                        concat=False,
                    )
                    if i in self.fusion_layers
                    else None,
                    # ("video", "video-audio", "audio"): GATConv((-1, -1), hidden_channels, add_self_loops=self_loops,
                    #                                            heads=1, concat=False),
                },
                aggr="sum",
            )

            self.convs.append(conv)
            self.Norm_layers.append(
                {
                    "audio": LayerNorm(hidden_channels).cuda(),
                    "video": LayerNorm(hidden_channels).cuda(),
                }
            )

        # # define readout layer
        # self.readout = GraphMultisetTransformer(in_channels=hidden_channels, out_channels=hidden_channels,
        #                                         hidden_channels=hidden_channels,
        #                                         pool_sequences=["GMPool_G"]
        #                                         )

        # define norm layers
        self.pre_norm_audio = nn.LayerNorm([self.num_aud_nodes, hidden_channels]).cuda()
        self.pre_norm_video = nn.LayerNorm([self.num_vid_nodes, hidden_channels]).cuda()

        # define output layer
        feat_dim = 2 * hidden_channels
        self.lin = nn.Sequential(
            nn.Linear(in_features=feat_dim, out_features=out_dim, bias=True)
        )
        # self.lin = nn.Sequential(
        #     nn.Linear(feat_dim, 1024),
        #     nn.ReLU(True),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(True),
        #     nn.Dropout(p=0.2),
        #     nn.Linear(1024, out_dim),
        # )

        # global attention pooling
        self.att_w_audio = nn.Linear(hidden_channels, 1, bias=False)
        xavier_uniform_(self.att_w_audio.weight)
        self.map_audio = nn.Identity()  # nn.Linear(hidden_channels, hidden_channels)
        self.graph_read_out_audio = GlobalAttention(
            gate_nn=self.att_w_audio, nn=self.map_audio
        )
        self.att_w_video = nn.Linear(hidden_channels, 1, bias=False)
        xavier_uniform_(self.att_w_video.weight)
        self.map_video = nn.Identity()  # nn.Linear(hidden_channels, hidden_channels)
        self.graph_read_out_video = GlobalAttention(
            gate_nn=self.att_w_video, nn=self.map_video
        )
        # attention weights for each sample
        self.att_w_audio_sample = None
        self.att_w_video_sample = None

        # Readout alternative
        # self.graph_read_out_audio = global_mean_pool
        # self.graph_read_out_video = global_mean_pool

    @classmethod
    def from_config(cls, cfg):

        return {
            "hidden_channels": cfg.MODEL.HIDDEN_CHANNELS,
            "out_dim": cfg.MODEL.OUT_DIM,
            "num_layers": cfg.MODEL.NUM_LAYERS,
            "device": cfg.DEVICE,
            "normalize": cfg.GRAPH.NORMALIZE,
            "self_loops": cfg.GRAPH.SELF_LOOPS,
            "vis_period": cfg.VIS_PERIOD,
            "loss": cfg.TRAINING.LOSS,
            "cfg": cfg,
        }

    def construct_matching_graph(
        self, x_dict, edge_index_dict, ptr_audio, ptr_video, batches
    ):
        """Construct the matching graph between modalities for the given input batch.
        Args:
            x_dict (dict): dictionary of input tensors
            edge_index_dict (dict): dictionary of edge indices
            ptr_audio (list): list of pointer to the audio graphs
            ptr_video (list): list of pointer to the video graphs
        Returns:
            dict: dictionary of input graphs edges plus the added matching edges
        """

        # get the input graphs features
        audio_feat = x_dict["audio"]
        video_feat = x_dict["video"]

        # dist = pairwise_distances(audio_graph, video_graph)
        # knn_audio = dist.topk(3, largest=False, dim=0).indices
        # knn_video = dist.topk(3, largest=False, dim=1).indices
        num_audio_graphs = num_of_graphs(ptr_audio)
        num_video_graphs = num_of_graphs(ptr_video)

        k = 3
        audio_video_edge_index = torch.zeros(
            [2, k * num_audio_graphs.sum()], dtype=torch.int64
        )
        video_audio_edge_index = torch.zeros(
            [2, k * num_video_graphs.sum()], dtype=torch.int64
        )
        target_nodes_video = []
        for i, (last_audio_idx, last_video_idx) in enumerate(zip(ptr_audio, ptr_video)):
            if i == 0:
                first_audio_idx = ptr_audio[0]
                first_video_idx = ptr_video[0]
            else:
                dist = pairwise_distances(
                    audio_feat[first_audio_idx:last_audio_idx],
                    video_feat[first_video_idx:last_video_idx],
                )
                knn_video = dist.topk(k, largest=False, dim=0).indices
                knn_audio = dist.topk(k, largest=False, dim=1).indices

                # target_nodes_video.append(knn_video.T.reshape(-1))

                for node_idx in range(num_video_graphs[i]):
                    first_node_idx = (ptr_video[i - 1] + node_idx) * k
                    second_node_idx = (ptr_video[i - 1] + node_idx + 1) * k
                    video_audio_edge_index[0, first_node_idx:second_node_idx] = (
                        node_idx + ptr_video[i - 1]
                    )
                    video_audio_edge_index[1, first_node_idx:second_node_idx] = (
                        knn_video[:, node_idx] + ptr_audio[i - 1]
                    )
                for node_idx in range(num_audio_graphs[i]):
                    first_node_idx = (ptr_audio[i - 1] + node_idx) * k
                    second_node_idx = (ptr_audio[i - 1] + node_idx + 1) * k
                    audio_video_edge_index[0, first_node_idx:second_node_idx] = (
                        node_idx + ptr_audio[i - 1]
                    )
                    audio_video_edge_index[1, first_node_idx:second_node_idx] = (
                        knn_audio[node_idx, :] + ptr_video[i - 1]
                    )

                first_audio_idx = last_audio_idx
                first_video_idx = last_video_idx
        # source_nodes_video = torch.from_numpy(
        #     np.array(list(range(num_video_graphs[1]))).reshape(1, -1).repeat(k)).repeat(
        #     len(num_video_graphs) - 1)
        # target_nodes_video = torch.cat(target_nodes_video, dim=0)

        edge_index_dict[("video", "video-audio", "audio")] = video_audio_edge_index.to(
            self.device
        )

        return edge_index_dict

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.
        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        ### Change here later
        a = 0
        # from detectron2.utils.visualizer import Visualizer
        #
        # storage = get_event_storage()
        # max_vis_prop = 20
        #
        # for input, prop in zip(batched_inputs, proposals):
        #     img = input["image"]
        #     img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
        #     v_gt = Visualizer(img, None)
        #     v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
        #     anno_img = v_gt.get_image()
        #     box_size = min(len(prop.proposal_boxes), max_vis_prop)
        #     v_pred = Visualizer(img, None)
        #     v_pred = v_pred.overlay_instances(
        #         boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
        #     )
        #     prop_img = v_pred.get_image()
        #     vis_img = np.concatenate((anno_img, prop_img), axis=1)
        #     vis_img = vis_img.transpose(2, 0, 1)
        #     vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
        #     storage.put_image(vis_name, vis_img)
        #     break  # only visualize one image in a batch

    def graph_forward(self, x_dict, batches, batched_edges, ptr_audio, ptr_video):

        # feeding audio and video features to the graph model
        edge_index_dict = batched_edges
        for i, (conv, norm_layer) in enumerate(zip(self.convs, self.Norm_layers)):
            if i in self.fusion_layers:
                edge_index_dict = self.construct_matching_graph(
                    x_dict, edge_index_dict, ptr_audio, ptr_video, batches
                )
            last_x_dict = x_dict
            x_dict, _ = conv(x_dict, edge_index_dict)
            x_dict = {key: x.relu() for key, x in x_dict.items()}
            if self.training:
                x_dict = {key: F.dropout(x, p=0.1) for key, x in x_dict.items()}
            x_dict = {
                key: norm_layer[key](x, batches[key]) for key, x in x_dict.items()
            }
            # residual connection
            if self.residual:
                x_dict = {key: x + last_x_dict[key] for key, x in x_dict.items()}

        # extracting the graph embeddings (merging audio and video embeddings)
        # graph_embed = self.graph_read_out_audio(x_dict["audio"], batches["audio"]) + \
        #               self.graph_read_out_video(x_dict["video"], batches["video"])
        graph_embed = torch.cat(
            [
                self.graph_read_out_audio(x_dict["audio"], batches["audio"]),
                self.graph_read_out_video(x_dict["video"], batches["video"]),
            ],
            dim=1,
        )
        # graph_embed = self.graph_read_out_audio(x_dict["audio"], batches["audio"])

        # bypassing the graph model
        graph_embed = torch.cat(
            [
                self.graph_read_out_audio(x_dict["audio"], batches["audio"]),
                self.graph_read_out_video(x_dict["video"], batches["video"]),
            ],
            dim=1,
        )

        return graph_embed

    def input_prep(self, audio, audio_fps, video, video_fps):
        # segments audio and video frames with overlapping windows
        audio_kernel_size = int(self.audio_seg_len * audio_fps * 0.001)
        audio_overlap = (audio.shape[2] - audio_kernel_size) // (self.num_aud_nodes - 1)
        audio = audio.unfold(dimension=2, size=audio_kernel_size, step=audio_overlap)
        video_kernel_size = int(self.video_seg_len * video_fps * 0.001)
        video_overlap = (video.shape[2] - video_kernel_size) // (self.num_vid_nodes - 1)
        video = video.unfold(dimension=2, size=video_kernel_size, step=video_overlap)

        # # segments audio and video frames without overlapping windows
        # batched_inputs["audio"] = torch.tensor_split(batched_inputs["audio"], num_audio_nodes, dim=1)
        # batched_inputs["video"] = torch.tensor_split(batched_inputs["video"], num_video_nodes, dim=1)

        return audio, video

    def audio_head_forward(self, audio):
        (
            a_B,
            a_C,
            a_S,
            a_L,
        ) = (
            audio.shape
        )  # a_B: batch size, a_C: channels, a_S: segments, a_L: length of waveform
        audio = audio.permute(0, 2, 1, 3)  # a_B, a_S, a_C, a_L
        audio = audio.mean(dim=2)  # convert to mono (a_B, a_S, a_L)
        audio = audio.reshape(a_B * a_S, a_L)
        # audio_feats = self.audio_head.extract_features(audio[:,0,0,:], num_layers=12)[0]
        # audio_feats = self.audio_head.feature_extractor(audio, length=None)[0]
        audio_feats = self.audio_head(audio, length=None)[0]
        audio_feats = audio_feats.mean(dim=1).view(a_B, a_S, -1)

        return audio_feats

    def video_head_forward(self, video):
        # placeholder for batch features
        features = {}

        # hook for extracting intermediate features
        def get_features(name):
            def hook(model, input, output):
                features[name] = output

            return hook

        self.video_head.avgpool.register_forward_hook(get_features("feats"))

        # feed video segments to the video model
        (
            v_B,
            v_C,
            v_S,
            v_H,
            v_W,
            v_T,
        ) = (
            video.shape
        )  # v_B: batch size, v_C: channels, v_S: segments, v_H: height, v_W: width, v_T: time,
        video = video.permute(
            0, 2, 1, 5, 3, 4
        ).contiguous()  # v_B, v_S, v_C, v_T, v_H, v_W
        video = video.view(v_B * v_S, v_C, v_T, v_H, v_W)  # v_B*v_S, v_C, v_T, v_H, v_W
        video_output = self.video_head(video)  # v_B*v_S, vid_feat_dim
        video_feats = features["feats"].view(v_B, v_S, -1)  # v_B, v_S, vid_feat_dim

        return video_feats

    def compute_loss_and_metrics(self, pred, batched_inputs):
        # compute loss
        classification_loss = {}
        if self.loss == "CrossEntropyLoss":
            criterion = torch.nn.CrossEntropyLoss()
            classification_loss["cls_loss"] = criterion(
                pred, batched_inputs["numerical_label"].to(self.device)
            )
        elif self.loss == "FocalLoss":
            classification_loss["cls_loss"] = focal_loss(
                pred,
                batched_inputs["numerical_label"].to(self.device),
                alpha=0.5,
                gamma=2.0,
                reduction="mean",
            )
        target = (
            F.one_hot(batched_inputs["numerical_label"], self.out_dim)
            .type(torch.FloatTensor)
            .to(self.device)
        )
        # classification_loss['cls_loss'] = F.binary_cross_entropy(pred, target)

        # computing average precision
        # try:
        #     average_precision = metrics.average_precision_score(
        #         target.cpu().float().numpy(),
        #         pred.detach().cpu().float().numpy(),
        #         average=None,
        #     )
        # except ValueError:
        #     average_precision = np.array([np.nan] * self.out_dim)
        average_precision = np.array([np.nan] * self.out_dim)
        # try:
        #     roc = metrics.roc_auc_score(batched_inputs['numerical_label'].numpy(), pred.softmax(1).numpy(),
        #     multi_class='ovr' )
        # except ValueError:
        #     roc = np.array([np.nan] * 527)

        # computing accuracy
        acc = (
            torch.max(pred, 1)[1].cpu() == batched_inputs["numerical_label"]
        ).sum().item() / batched_inputs["numerical_label"].shape[0]

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs)

        losses = {}
        losses.update(classification_loss)
        # losses.update(proposal_losses)
        metric = {}
        metric["mAP"] = torch.from_numpy(
            np.asarray(
                average_precision[~np.isnan(average_precision)].sum()
                / average_precision.shape[0]
            )
        ).to(self.device)
        metric["accuracy"] = torch.from_numpy(np.asarray(acc)).to(self.device)

        return losses, metric

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        batched_inputs["graph"] = batched_inputs["graph"].to(self.device)
        ptr_audio = batched_inputs["graph"]["audio"]["ptr"]
        ptr_video = batched_inputs["graph"]["video"]["ptr"]

        # read graph data
        batched_x, batched_edges, batches = (
            batched_inputs["graph"].x_dict,
            batched_inputs["graph"].edge_index_dict,
            batched_inputs["graph"].batch_dict,
        )
        num_audio_nodes = int(batched_inputs["num_aud_nodes"][0])
        audio_fps = int(batched_inputs["audio_fps"][0])
        num_video_nodes = int(batched_inputs["num_vid_nodes"][0])
        video_fps = int(batched_inputs["video_fps"][0])

        # check the consistency between input number of nodes with the model parameters
        assert num_audio_nodes == self.num_aud_nodes
        assert num_video_nodes == self.num_vid_nodes

        # read audio and video frames
        batched_inputs["audio"] = batched_inputs["audio"].to(self.device)
        batched_inputs["video"] = batched_inputs["video"].to(self.device)

        # prepare input for graph model
        audio, video = self.input_prep(
            batched_inputs["audio"], audio_fps, batched_inputs["video"], video_fps
        )

        # feed audio segments to the audio model
        audio_feats = self.audio_head_forward(audio)

        # feed video frames to the video model
        video_feats = self.video_head_forward(video)

        # apply norm layers
        # audio_feats = self.pre_norm_audio(audio_feats)
        # video_feats = self.pre_norm_video(video_feats)

        # update graph data
        batched_x["audio"] = audio_feats.flatten(start_dim=0, end_dim=1)
        batched_x["video"] = video_feats.flatten(start_dim=0, end_dim=1)

        # computing MAD metric
        # MAD_audio_begin = np.mean(1 - cosine_similarity(x_dict["audio"].detach().cpu().numpy(),
        #                                               x_dict["audio"].detach().cpu().numpy()))
        # MAD_video_begin = np.mean(1 - cosine_similarity(x_dict["video"].detach().cpu().numpy(),
        #                                               x_dict["video"].detach().cpu().numpy()))

        # apply graph model
        graph_embed = self.graph_forward(
            x_dict=batched_x,
            batches=batches,
            batched_edges=batched_edges,
            ptr_audio=ptr_audio,
            ptr_video=ptr_video,
        )

        # pred = self.lin(torch.sigmoid(graph_embed))
        pred = self.lin(graph_embed)

        losses, metric = self.compute_loss_and_metrics(pred, batched_inputs)

        # computing MAD metric
        # MAD_audio_end = np.mean(1 - cosine_similarity(x_dict["audio"].detach().cpu().numpy(),
        #                               x_dict["audio"].detach().cpu().numpy()))
        # MAD_video_end = np.mean(1 - cosine_similarity(x_dict["video"].detach().cpu().numpy(),
        #                               x_dict["video"].detach().cpu().numpy()))
        # metric["MAD_audio_begin"] = torch.from_numpy(np.asarray(MAD_audio_begin)).to(self.device)
        # metric["MAD_video_begin"] = torch.from_numpy(np.asarray(MAD_video_begin)).to(self.device)
        # metric['MAD_audio_end'] = torch.from_numpy(np.asarray(MAD_audio_end)).to(self.device)
        # metric['MAD_video_end'] = torch.from_numpy(np.asarray(MAD_video_end)).to(self.device)
        # metric["attn_div_score_GAT"] = atten_dive_score(_[1], _[0])
        # metric["attn_div_score_audio"] = atten_dive_score(self.att_w_audio_sample)
        # metric["attn_div_score_video"] = atten_dive_score(self.att_w_video_sample)
        # losses["cls_loss"] = losses["cls_loss"] - metric["attn_div_score_GAT"] * 0.1 \
        #                     - metric["attn_div_score_audio"] * 0.1 \
        #                     - metric["attn_div_score_video"] * 0.1

        return losses, metric

    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.
        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        batched_inputs["graph"] = batched_inputs["graph"].to(self.device)
        ptr_audio = batched_inputs["graph"]["audio"]["ptr"]
        ptr_video = batched_inputs["graph"]["video"]["ptr"]

        # read graph data
        batched_x, batched_edges, batches = (
            batched_inputs["graph"].x_dict,
            batched_inputs["graph"].edge_index_dict,
            batched_inputs["graph"].batch_dict,
        )
        num_audio_nodes = int(batched_inputs["num_aud_nodes"][0])
        audio_fps = int(batched_inputs["audio_fps"][0])
        num_video_nodes = int(batched_inputs["num_vid_nodes"][0])
        video_fps = int(batched_inputs["video_fps"][0])

        # read audio and video frames
        batched_inputs["audio"] = batched_inputs["audio"].to(self.device)
        batched_inputs["video"] = batched_inputs["video"].to(self.device)

        # prepare input for graph model
        audio, video = self.input_prep(
            batched_inputs["audio"], audio_fps, batched_inputs["video"], video_fps
        )

        # feed audio segments to the audio model
        audio_feats = self.audio_head_forward(audio)

        # feed video frames to the video model
        video_feats = self.video_head_forward(video)

        # apply norm layers
        audio_feats = self.pre_norm_audio(audio_feats)
        video_feats = self.pre_norm_video(video_feats)

        # update graph data
        batched_x["audio"] = audio_feats.flatten(start_dim=0, end_dim=1)
        batched_x["video"] = video_feats.flatten(start_dim=0, end_dim=1)

        graph_embed = self.graph_forward(
            x_dict=batched_x,
            batches=batches,
            batched_edges=batched_edges,
            ptr_audio=ptr_audio,
            ptr_video=ptr_video,
        )

        # pred = self.lin(torch.sigmoid(graph_embed))
        pred = self.lin(graph_embed)

        # from torch_geometric.utils import softmax
        # self.att_w_audio_sample = softmax(
        #     self.graph_read_out_audio.gate_nn(x_dict["audio"]).view(-1, 1), batches["audio"],
        #     num_nodes=batches["audio"][-1]+1
        # ).view(1, -1)
        # self.att_w_audio_sample = torch.cat(
        #     [
        #         self.att_w_audio_sample[:, batches["audio"] == i]
        #         for i in torch.unique(batches["audio"])
        #     ]
        # )
        # self.att_w_video_sample = softmax(
        #     self.graph_read_out_video.gate_nn(x_dict["video"]).view(-1, 1), batches["video"],
        #     num_nodes=batches["video"][-1]+1
        # ).view(1, -1)
        # self.att_w_video_sample = torch.cat(
        #     [
        #         self.att_w_video_sample[:, batches["video"] == i]
        #         for i in torch.unique(batches["video"])
        #     ]
        # )

        # for i, (attn_a, attn_v) in enumerate(zip(self.att_w_audio_sample, self.att_w_video_sample)):
        #     if '19730' in batched_inputs['video_add'][i]:
        #         if torch.where(attn_v > 0.03)[0].shape[0] > 0 or torch.where(attn_a > 0.02)[0].shape[0] > 0:
        #             # print(i)
        #             plt.figure()
        #             v = self.att_w_video_sample[i].detach().cpu().numpy()
        #             v = (v - np.min(v)) / (np.max(v) - np.min(v))
        #             a = self.att_w_audio_sample[i].detach().cpu().numpy()
        #             a = (a - np.min(a)) / (np.max(a) - np.min(a))
        #             plt.plot(signal.resample(savgol_filter(v, 39, 3), 101))
        #             plt.plot(savgol_filter(a, 21, 3))
        #             plt.legend(["video", "audio"])
        #             plt.xlabel("Graph Nodes")
        #             plt.ylabel("Attention Weight")
        #             plt.show()
        #             plt.close()

        return pred.detach().cpu()
