from graphviz import Source
dot_script = r"""
digraph G {
    compound=true;        // 允许集群之间的边
    nodesep=0.6;          // 节点间水平间距
    ranksep=0.8;          // 节点间垂直间距 (rank)
    fontname="Arial";     // 全局字体
    splines=ortho;        // 使用正交（直角）连接线，这对于框图至关重要
    rankdir=TB;           // 默认布局从上到下

    // 默认节点样式
    node [shape=box, style=filled, fontname="Arial", margin="0.25,0.15"];
    // 默认边样式
    edge [fontname="Arial", penwidth=1.5];

    // --- (a) 部分: 主干网络和检测头 ---
    // (a) 的标签将作为集群标签
    subgraph cluster_a {
        label="(a)";
        labelloc=t;
        labeljust=l;
        fontsize=16;
        style=invis; // 隐藏集群边框

        // --- (a) 主干 (Backbone) ---
        subgraph cluster_a_backbone {
            style=invis; // 隐藏集群边框
            rankdir=TB;

            node_image [label="Image", style="filled,rounded", fillcolor="#FFFFFF", color="black"];
            node_patch_part [label="Patch Partition", fillcolor="#4FCAD6"];
            node_linear_emb [label="Linear Embedding", fillcolor="#4FCAD6"];
            node_swin1 [label="Swin Transformer\nBlock", fillcolor="#4FCAD6"];
            node_patch_merge1 [label="Patch Merging", fillcolor="#4FCAD6"];
            node_swin2 [label="Swin Transformer\nBlock", fillcolor="#4FCAD6"];
            node_patch_merge2 [label="Patch Merging", fillcolor="#4FCAD6"];
            node_swin3 [label="Swin Transformer\nBlock", fillcolor="#4FCAD6"];
            node_patch_merge3 [label="Patch Merging", fillcolor="#4FCAD6"];
            node_swin4 [label="Swin Transformer\nBlock", fillcolor="#4FCAD6"];
            node_sppf_main [label="SPPF", fillcolor="#F7A94E"];

            // 垂直连接主干
            node_image -> node_patch_part -> node_linear_emb -> node_swin1 -> node_patch_merge1 -> node_swin2 -> node_patch_merge2 -> node_swin3 -> node_patch_merge3 -> node_swin4 -> node_sppf_main [penwidth=2];
        }

        // --- (a) 颈部和头部 (Neck & Head) ---
        // 节点定义
        
        // 顶部分支 (P3)
        node_csp1 [label="CSP", fillcolor="#88C3E8"];
        node_conv_block1_1 [label="Conv Block", fillcolor="#D1E8B4"];
        node_conv_block1_2 [label="Conv Block", fillcolor="#D1E8B4"];
        node_conv2d1_1 [label="Conv2D", fillcolor="#D1E8B4"];
        node_conv2d1_2 [label="Conv2D", fillcolor="#D1E8B4"];
        node_bbox_loss1 [label="BBox Loss", fillcolor="#E0D4C5"];
        node_cls_loss1 [label="Cls. Loss", fillcolor="#E0D4C5"];

        // 中间分支 (P4)
        node_upsample1 [label="UpSample", fillcolor="#B4D9EB"];
        node_concat1 [label="Concat", fillcolor="#F7A9C2"];
        node_csp2 [label="CSP", fillcolor="#88C3E8"];
        node_conv_block2_1 [label="Conv Block", fillcolor="#D1E8B4"];
        node_conv_block2_2 [label="Conv Block", fillcolor="#D1E8B4"];
        node_conv2d2_1 [label="Conv2D", fillcolor="#D1E8B4"];
        node_conv2d2_2 [label="Conv2D", fillcolor="#D1E8B4"];
        node_bbox_loss2 [label="BBox Loss", fillcolor="#E0D4C5"];
        node_cls_loss2 [label="Cls. Loss", fillcolor="#E0D4C5"];
        
        // 底部分支 (P5)
        node_upsample2 [label="UpSample", fillcolor="#B4D9EB"];
        node_concat2 [label="Concat", fillcolor="#F7A9C2"];
        node_csp3 [label="CSP", fillcolor="#88C3E8"];
        node_conv_block3_1 [label="Conv Block", fillcolor="#D1E8B4"];
        node_conv_block3_2 [label="Conv Block", fillcolor="#D1E8B4"];
        node_conv2d3_1 [label="Conv2D", fillcolor="#D1E8B4"];
        node_conv2d3_2 [label="Conv2D", fillcolor="#D1E8B4"];
        node_bbox_loss3 [label="BBox Loss", fillcolor="#E0D4C5"];
        node_cls_loss3 [label="Cls. Loss", fillcolor="#E0D4C5"];

        // 连接 (a) 的颈部和头部
        // FPN-like 上采样路径
        node_swin2 -> node_csp1 [minlen=4];       // 从主干引出
        node_swin3 -> node_upsample1 [minlen=4];  // 从主干引出
        node_sppf_main -> node_upsample2 [minlen=4]; // 从主干引出

        node_upsample1 -> node_concat1;
        node_csp1 -> node_concat1 [minlen=2]; // 跨层连接
        
        node_upsample2 -> node_concat2;
        node_csp2 -> node_concat2 [minlen=2]; // 跨层连接
        
        node_concat1 -> node_csp2;
        node_concat2 -> node_csp3;

        // 检测头分支
        node_csp1 -> node_conv_block1_1 -> node_conv2d1_1 -> node_bbox_loss1;
        node_csp1 -> node_conv_block1_2 -> node_conv2d1_2 -> node_cls_loss1;

        node_csp2 -> node_conv_block2_1 -> node_conv2d2_1 -> node_bbox_loss2;
        node_csp2 -> node_conv_block2_2 -> node_conv2d2_2 -> node_cls_loss2;

        node_csp3 -> node_conv_block3_1 -> node_conv2d3_1 -> node_bbox_loss3;
        node_csp3 -> node_conv_block3_2 -> node_conv2d3_2 -> node_cls_loss3;

        // --- (a) 布局对齐 ---
        // 使用 rank=same 强制水平对齐
        { rank=same; node_swin2; node_csp1; }
        { rank=same; node_swin3; node_upsample1; node_concat1; }
        { rank=same; node_patch_merge3; node_csp2; } // 尝试对齐
        { rank=same; node_sppf_main; node_upsample2; node_concat2; }
        { rank=same; node_swin4; node_csp3; } // 尝试对齐

        { rank=same; node_conv_block1_1; node_conv_block1_2; node_conv_block2_1; node_conv_block2_2; node_conv_block3_1; node_conv_block3_2; }
        { rank=same; node_conv2d1_1; node_conv2d1_2; node_conv2d2_1; node_conv2d2_2; node_conv2d3_1; node_conv2d3_2; }
        { rank=same; node_bbox_loss1; node_cls_loss1; node_bbox_loss2; node_cls_loss2; node_bbox_loss3; node_cls_loss3; }
    }


    // --- (b) Swin Transformer Block 定义 ---
    subgraph cluster_b {
        label="(b)";
        labelloc=t;
        labeljust=l;
        fontsize=16;
        style=dashed; // 使用虚线框
        rankdir=LR;   // 此子图从左到右

        // 定义 + 节点
        node [shape=circle, fixedsize=true, width=0.4, fillcolor="#D1E8B4"];
        b_add1 [label="+"];
        b_add2 [label="+"];
        b_add3 [label="+"];
        b_add4 [label="+"];
        
        // 定义其他节点
        node [shape=box, fixedsize=false, fillcolor="#B4D9EB"];
        b_ln1 [label="LN"];
        b_wmsa [label="W-MSA"];
        b_ln2 [label="LN"];
        b_swmsa [label="SW-MSA"];
        node [fillcolor="#F7A9C2"];
        b_mlp1 [label="MLP"];
        b_mlp2 [label="MLP"];
        
        // 定义不可见的输入/输出点
        node [shape=point, width=0.01, height=0.01, style=invis];
        b_in;
        b_mid;
        b_out;

        // 连接 W-MSA 块
        b_in -> b_ln1 [penwidth=2];
        b_ln1 -> b_wmsa -> b_add1 [penwidth=2];
        b_in -> b_add1 [constraint=false]; // 残差连接
        b_add1 -> b_mlp1 -> b_add2 [penwidth=2];
        b_add1 -> b_add2 [constraint=false]; // 残差连接
        b_add2 -> b_mid [penwidth=2];

        // 连接 SW-MSA 块
        b_mid -> b_ln2 [penwidth=2];
        b_ln2 -> b_swmsa -> b_add3 [penwidth=2];
        b_mid -> b_add3 [constraint=false]; // 残差连接
        b_add3 -> b_mlp2 -> b_add4 [penwidth=2];
        b_add3 -> b_add4 [constraint=false]; // 残差连接
        b_add4 -> b_out [penwidth=2];

        // 对齐 (b)
        { rank=same; b_in; b_mid; }
        { rank=same; b_ln1; b_ln2; }
        { rank=same; b_wmsa; b_swmsa; }
        { rank=same; b_add1; b_add3; }
        { rank=same; b_mlp1; b_mlp2; }
        { rank=same; b_add2; b_add4; }
    }

    // --- (c) 和 (d) 的容器 ---
    // 我们将 (c) 和 (d) 放在一个不可见的集群中，以将它们与 (a) 分开
    subgraph cluster_bottom {
        style=invis;
        rankdir=LR; // (c) 和 (d) 左右并排

        // --- (c) 组件定义 ---
        subgraph cluster_c {
            label="(c)";
            labelloc=t;
            labeljust=l;
            fontsize=16;
            style=dashed; // 虚线框
            rankdir=LR;   // 此子图从左到右

            // Conv Block
            c_conv_eq [label="Conv Block      =", fillcolor="#D1E8B4"];
            c_conv2d [label="Conv2D", fillcolor="#D1E8B4"];
            c_bn [label="BatchNorm2d", fillcolor="#D1E8B4"];
            c_silu [label="SILU", fillcolor="#D1E8B4"];
            c_conv_eq -> c_conv2d -> c_bn -> c_silu [penwidth=2];

            // CSP
            c_csp_eq [label="CSP                   =", fillcolor="#88C3E8"];
            c_csp_conv1 [label="Conv Block", fillcolor="#D1E8B4"];
            c_split [label="Split", fillcolor="#B4D9EB"];
            c_csp_conv2 [label="Conv Block", fillcolor="#D1E8B4"];
            c_bottle [label="Bottleneck", fillcolor="#F7A9C2"];
            c_concat [label="Concat", fillcolor="#F7A9C2"];
            c_csp_conv3 [label="Conv Block", fillcolor="#D1E8B4"];
            
            c_csp_eq -> c_csp_conv1 -> c_split [penwidth=2];
            c_split -> c_csp_conv2 [penwidth=2];
            c_split -> c_bottle [penwidth=2];
            c_csp_conv2 -> c_concat [penwidth=2];
            c_bottle -> c_concat [penwidth=2];
            c_concat -> c_csp_conv3 [penwidth=2];
            { rank=same; c_csp_conv2; c_bottle; } // 水平对齐

            // Bottleneck
            c_bottle_eq [label="Bottleneck      =", fillcolor="#F7A9C2"];
            c_bottle_conv1 [label="Conv Block", fillcolor="#D1E8B4"];
            c_bottle_conv2 [label="Conv Block", fillcolor="#D1E8B4"];
            c_add [label="+", shape=circle, fixedsize=true, width=0.4, fillcolor="#D1E8B4"];
            c_invis [shape=point, width=0.01, height=0.01, style=invis]; // 用于残差连接的起点

            c_bottle_eq -> c_invis [style=invis];
            c_invis -> c_bottle_conv1 -> c_bottle_conv2 -> c_add [penwidth=2];
            c_invis -> c_add [constraint=false]; // 残差连接
            { rank=same; c_bottle_eq; c_invis; } // 对齐

            // 对齐 (c) 中的 = 
            { rank=min; c_conv_eq; c_csp_eq; c_bottle_eq; }
        }

        // --- (d) SPPF 定义 ---
        subgraph cluster_d {
            label="(d)";
            labelloc=t;
            labeljust=l;
            fontsize=16;
            style=dashed; // 虚线框
            rankdir=LR;   // 此子图从左到右

            d_sppf_eq [label="SPPF                =", fillcolor="#F7A94E"];
            d_conv1 [label="Conv Block", fillcolor="#D1E8B4"];
            d_pool1 [label="MaxPool2d", fillcolor="#D1E8B4"];
            d_pool2 [label="MaxPool2d", fillcolor="#D1E8B4"];
            d_pool3 [label="MaxPool2d", fillcolor="#D1E8B4"];
            d_concat [label="Concat", fillcolor="#F7A9C2"];
            d_conv2 [label="Conv Block", fillcolor="#D1E8B4"];

            d_sppf_eq -> d_conv1 [penwidth=2];
            d_conv1 -> d_pool1 [penwidth=2];
            d_conv1 -> d_pool2 [penwidth=2];
            d_conv1 -> d_pool3 [penwidth=2];
            d_pool1 -> d_concat [penwidth=2];
            d_pool2 -> d_concat [penwidth=2];
            d_pool3 -> d_concat [penwidth=2];
            d_concat -> d_conv2 [penwidth=2];

            // 对齐 (d)
            { rank=same; d_pool1; d_pool2; d_pool3; } // 水平对齐
            { rank=min; d_sppf_eq; }
        }
    }
    
    // --- 最终布局 ---
    // 强制 (b) 在 (a) 的右侧
    node_swin1 -> b_in [style=invis, minlen=8];
    // 强制 (c) 和 (d) 在 (a) 的下方
    node_sppf_main -> c_conv_eq [style=invis, minlen=4];
}
"""

src = Source(dot_script, filename="network_architecture", format="png")
src.render(cleanup=True)  # cleanup=True 删除临时文件

print("生成完成，图片保存为 network_architecture.png")