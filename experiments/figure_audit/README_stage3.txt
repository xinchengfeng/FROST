FROST 稿件修订：第三阶段

本阶段完成软件证据复核、七张图重绘、全文与补充材料的图件替换。
第二阶段的科学内容和已有数值保留；新增了复核说明与更清楚的图注。
本阶段执行冻结算子的推理及采样重放、已有重构输出的独立评分，未重新训练模型。

在 Overleaf 使用
1. 上传 FROST_NC_stage3_Overleaf.zip 中的全部内容。
2. 将 FROST_NC_stage3_main.tex 设为主文件，使用 pdfLaTeX 编译。
3. 补充材料单独以 FROST_NC_stage3_Supplementary.tex 为主文件编译。
4. FROST_references_NC.bib 与 FROST_nature.bst 已包括在包内。
5. Images 中七个 FROST_*.pdf 是本阶段的新图；其余图为保留的原有图件。
6. 本地可运行：
   latexmk -pdf -interaction=nonstopmode -halt-on-error FROST_NC_stage3_main.tex
   latexmk -pdf -interaction=nonstopmode -halt-on-error FROST_NC_stage3_Supplementary.tex

图件的位置
FROST_execution.pdf：Results 首个小节，实际 pre-normalization 执行图。
FROST_dimension_transfer.pdf：维度与分布迁移，a,b 在上、c 通栏在下。
FROST_sampler_precision.pdf：CDF 位宽与随机源，a,b 在上、c 通栏在下。
FROST_attention_convergence.pdf：收敛小节，误差地图在上、误差曲线在下。
FROST_BSDS500_distribution.pdf：重构小节，完整 200-image 分布及配对比较。
FROST_reconstruction.pdf：重构小节，原有三个 pilot 的图像、ROI 与误差。
FROST_attention_formation.pdf：Supplementary Fig. S2，FSM 轨迹与概率单纯形。

单独复制代码时，FROST_stage3_main_figures.tex 和
FROST_stage3_supplement_figure.tex 提供对应 figure 环境。
请替换现有同名图环境，避免重复插入及 label 重复。
本稿在导言区使用 caption 包将图注统一为 9 pt：
\usepackage[font=footnotesize,labelfont=bf]{caption}

重现图件
FROST_NC_stage3_SourceData.zip 为可独立重画本阶段图件的轻量包，包含：
- 全部新图使用的 CSV/JSON 汇总与核验报告；
- 16 条原始收敛轨迹和固定输入、原有三个 pilot 的原始输出；
- 条件分类分布、维度/采样及全部 BSDS500 逐图逐流分数；
- plot_stage3.py、formation_layout.py、audit_vector_assets.py、字体。
解压后在该目录运行：
   python source/plot_stage3.py
   python source/audit_vector_assets.py
绘图脚本仅依赖 NumPy、Matplotlib；结构核验另需 pypdf 和 Poppler 的 pdffonts。
requirements-stage3.txt 记录本地完整复核环境的版本，包含 PyTorch CPU 构建。
无需重新训练或下载整个图像数据集即可重新绘制全部七张图。

重新执行完整证据复核
verify_stage3.py 也随包提供，但完整复核需要前面已经保存的原始资产，
它们没有在这个轻量绘图包中再次复制。请用参数指向原有三个资产目录：
   python source/verify_stage3.py --revision REVISION_DIR --bsds BSDS_DIR \
      --convergence CONVERGENCE_DIR --out NEW_VERIFICATION_DIR
REVISION_DIR 包含 source/、checkpoints/ 和 source_data/；
BSDS_DIR 包含 data/test/、runs/ 和 400 个检查点；
CONVERGENCE_DIR 包含 source_data/trace_*.npz。
这些参数均是本地路径，复核脚本不会修改输入目录。
本阶段完整复核实际已执行，详细报告位于 source_data/*_audit.json。

数据解释
- L 最大为 4096，注意力采样另有 256 burn-in cycles。
- 维度实验的三个拟合共享 resized warm start；其标准差按拟合计算。
- 收敛与 CDF 图的误差带按八个固定权重下的 stream-level 均值计算。
- BSDS500 的误差项按 200 个 image-level 均值计算；三个流先在图内平均。
- BSDS500 各模型使用自己目标图像的全部像素拟合，是 fitted-image
  reconstruction，不是 held-out restoration、compression 或图像生成。
- 未删除四个 FROST 图像均值低于 30 dB 的目标，也未删除唯一负配对增益。
- 六张图是完全矢量；重构图包含真实 64x64 图像及 16x16 ROI 的像素，
  文字、边框和色条为矢量。没有对像素进行生成式增强或人为细节补全。
- 分类及检索的完整测试结果仍为归档数据，本阶段没有重新测定。
- 未进行任何硬件实验，也没有生成面积、功耗、速度的填充数值。

第四阶段仍负责最终全文排版、交叉引用和投稿前整体一致性检查。
本阶段提供的完整源码可以直接编译，亦可作为第四阶段的唯一工作基线。
