FROST: BSDS500 全量评估图像池的逐图坐标重建

评估范围与统计单位
使用官方测试分区的全部200张图片，无按质量排除。BSDS500共有500张图片；本次评估图像池是其中的200张测试分区照片，并非对500张图片全部拟合。
每张照片先将短边bicubic缩放到96像素，再取64×64 RGB中心裁剪。全部目标像素用于拟合该图自己的Transformer；不存在独立的像素测试集或图片级泛化评估。
每图使用一个固定训练初始化（11+图像ID），拟合1500步。原三张图的既有模型和算子沿用；其余197张严格使用相同架构和预算。额外位置嵌入冻结为零，推理只接收50维坐标/Fourier特征，目标信息位于拟合权重中。
每张图共享同一组拟合权重进行Float、W8A8、SC和FROST转换评估。SC在L=4096评估；FROST在L=1024和4096评估；两种长度均使用三个base stream seeds：101、202、303。
先给每张实际输出计算PSNR和SSIM，再对每图三个得分取均值，最后统计200个图像均值的总体均值和样本标准差。没有通过平均输出图像提高得分。跨图片的SD与同一拟合模型的随机流SD分别保存。600个流输出没有作为600张独立图像。

实测结果（mean ± sample SD，跨200个图像级得分）
Float, L=not applicable: PSNR 46.042844 ± 2.015891 dB; SSIM 0.99503593 ± 0.00330528; image mean PSNR≥30: 200/200.
W8A8, L=not applicable: PSNR 44.434841 ± 1.905566 dB; SSIM 0.99401893 ± 0.00345358; image mean PSNR≥30: 200/200.
SC_base, L=4096: PSNR 32.626823 ± 3.030911 dB; SSIM 0.95405306 ± 0.03373517; image mean PSNR≥30: 162/200.
FROST, L=1024: PSNR 32.751238 ± 2.252620 dB; SSIM 0.95512456 ± 0.01592497; image mean PSNR≥30: 177/200.
FROST, L=4096: PSNR 35.182381 ± 2.493592 dB; SSIM 0.97292406 ± 0.01349041; image mean PSNR≥30: 196/200.
FROST_analytic, L=not applicable: PSNR 36.452225 ± 2.735291 dB; SSIM 0.97895839 ± 0.01267800; image mean PSNR≥30: 198/200.

FROST L=4096: 588/600 individual stream outputs reach30dB; minimum run PSNR 28.38473049 dB. All outputs, including those below30dB, are retained.
FROST_analytic是数值稳态/期望控制，不属于有限比特流实验；它单独保存在源数据中。
原三张可视化图中的数字仍是base seed101的单次结果。新主表是200张图片的统计，误差项不再是原三图的种子波动。

实现及验证
SC/FROST的有限流输出来自真实Sobol数字移位比特、精确AND/popcount、饱和FSM轨迹及TT类别发射。C++循环仅加速这些整数操作，未使用期望替代或加性噪声近似。
exact_kernel_verification.json包含21项核验：L=64/256的整数内核、FSM和TT输出与原实现完全一致；原三图L=4096/seed101的FROST、SC及W8A8图像逐元素完全一致。
results_verification.json从200个实测npz中重新计算了2400个输出的PSNR/SSIM，并验证照片SHA256、参考裁剪以及原三图锚定结果。
W8A8控制量化线性投影、QK的操作数，以及用于AV的注意力权重和value。注意力权重使用255级无符号量化；其余量化操作数是有符号8位。归一化、softmax和GELU是数值操作。旧段落“仅量化线性投影”的表述已更正。
FROST的坐标编码、排序、均值/范围处理、残差/偏置等仍是数值操作。没有测量硬件面积、能量或延迟优势。
算子校准使用每图自身浮点模型的激活行80/20划分，在固定候选bound=2/4/8间选择，600步/候选；这是行级选择，而不是独立图片验证。

文件
FROST_BSDS500_reconstruction.tex：直接替换用户提供的整个段落、表格和图注；继续引用Images/FROST_reconstruction.pdf。依赖graphicx和booktabs。
figures/FROST_reconstruction.pdf：原三图可视化，保持原有实测图像和标注。
figures/FROST_BSDS500_distribution.pdf/svg/png：新增200张图像的PSNR分布与SC/FROST配对图。字体沿用Nimbus Sans，青绿/蓝/橙配色与前图一致。
FROST_BSDS500_distribution_insert.tex：可选分布图的LaTeX图注。
source_data/all_metrics.csv：2400条单次输出指标，包含所有种子、方法及长度。
source_data/per_image_metrics.csv：每图每方法的种子均值及种子SD，共1200条记录。
source_data/dataset_summary.csv/json：全量均值、跨图SD、中位数、低分位、最小值、配对增益及95%图像bootstrap区间。
source_data/protocol.json：固定协议、200个图像ID、源JPEG哈希和官方归档SHA256。
checkpoints/：每图的最佳Float拟合模型及校准算子；训练优化器resume文件不包含在最终包中。
runs/image_ID.npz/json：参考和全部实际输出/指标；三个旧结果锚定文件也保留。
FROST_references.bib：沿用已核验引用，包含BSDS500、Fourier features、AdamW和SSIM。

复现（解压后在包根目录运行；Python3.12、C++17编译器；版本记录见source_data/environment.json）
pip install -r requirements.txt
python source/scripts/fast_finite.py
python source/scripts/verify_fast_finite.py .
python source/scripts/evaluate_bsds500.py --root . --workers 8
python source/scripts/summarize_bsds500.py --root .
python source/scripts/write_manuscript.py --root .
python source/scripts/plot_bsds500.py --root .
已有完整实测文件会跳过；若需重新测量，复制data/、source/、fonts/和checkpoints/到新目录，并复制source_data/protocol.json及exact_kernel_verification.json，并将原三个reference_previous_ID.npz锚定文件复制到runs/；不复制image_ID输出。
若需从头拟合，先建立新实验目录，复制data/、source/、fonts/和source_data/protocol.json，checkpoints/保持为空，然后运行：
python source/scripts/benchmark_bsds500.py --root . --workers 7
重新拟合后的权重存在版本相关的数值差异时，应重新运行原始/加速实现核验，而不是要求它等于旧权重的锚定图像。随附权重与输出是本次论文数字的直接证据。

数据来源：https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz
参考：Arbelaez et al., Contour Detection and Hierarchical Image Segmentation, IEEE TPAMI 33, 898–916 (2011).
结果范围：逐图拟合重建及随机转换保真度；推理输入是坐标，不能据此写成未知图片恢复、无条件生成或压缩评估。
