# CVAT / FiftyOne 视频复核集成

本轮提供独立的离线 CLI，不增加 Dashboard 页面或自动标注服务。已有视频可先制作
Argus 复核包，送入 CVAT 人工复核，导出新的回归标签，再在 FiftyOne 中查看标签、
模型预测及错误案例。集成不会批准模型、改变发布状态或代替原有回归验收。

当前没有测试机；下文安装、CLI、测试和回归命令是测试环境恢复后的待验收步骤，
本轮没有执行。不要据此认定 CVAT 服务、FiftyOne 数据库或视频回归已经验证通过。

## 安装边界

CVAT SDK 使用可选依赖 `cvat`；包含两种工具的可选依赖名为 `review`。
普通 Argus 进程不会导入这些 SDK 或连接外部服务。SDK 适配以 CVAT 2.79 系列为目标，
`cvat-sdk>=2.79,<2.80`；服务版本升级时需重新验证。FiftyOne 的可选范围为 `>=1.8,<2`，
图片导入依赖该版本起提供的禁用单例缓存能力，不支持缺少此能力的旧 SDK。
这些范围是代码兼容目标，不是已经完成的部署认证；首次验收后应保存实际解析的依赖清单。

已有 Argus 开发环境中，只需要 CVAT 时可在测试机执行：

```powershell
python -m pip install -e ".[cvat]"
```

FiftyOne 建议使用独立环境，避免其数据库、数值库及视频依赖与 GPU 推理环境互相影响。
以下示例从仓库根目录运行，只把 Argus 源码加入路径，不安装整套生产推理依赖：

```powershell
python -m venv .venv-review
.\.venv-review\Scripts\Activate.ps1
python -m pip install "fiftyone>=1.8,<2" "pydantic>=2,<3"
$env:PYTHONPATH = (Resolve-Path .\src).Path
```

如果同一独立环境还需要制作复核包和操作 CVAT，再安装：

```powershell
python -m pip install "cvat-sdk>=2.79,<2.80" "PyYAML>=6,<7" "av>=12.0.0"
```

FiftyOne 自身使用数据库服务；启动应用前按上游安装要求确认操作系统和 MongoDB 支持。
复核机器必须能读取 bundle 内的源视频路径。首次实现使用绝对路径引用视频，不复制、
转码或上传副本到 FiftyOne；跨机器迁移应重新准备复核包，并核对源视频摘要。

## 制作复核包

输入复用 `scripts/run_regression_suite.py` 的 manifest。示例：

```yaml
version: 1
suite_id: inspection-2026-10
default_camera_id: camera-01
clips:
  - id: normal-01
    kind: negative
    video: videos/normal-01.mp4
    source_session_id: capture-2026-10-01-morning
  - id: foreign-01
    kind: static_foreign
    video: videos/foreign-01.mp4
    annotations: labels/foreign-01.labels.json
    source_session_id: capture-2026-10-02-afternoon
    scene_reference: references/camera-01.jpg
    expected_detection_types: [persistent_change]
```

路径以输入 manifest 所在目录为基准。`source_session_id` 表示完整采集批次，
包括同一批次的不同摄像头、相邻片段。可以在顶层用 `default_source_session_id`
提供共同批次，逐条声明时以 clip 为准。

可选 split 文件是 JSON 对象，取值为 `train`、`validation`、`test` 或 `unassigned`：

```json
{
  "normal-01": "validation",
  "foreign-01": "test"
}
```

没有分配时保持 `unassigned`。分配到训练/验证/测试的片段必须提供采集批次；
相同视频 SHA256 或相同采集批次跨集合会被拒绝。不要通过更换批次名字拆分相邻帧。

```powershell
python scripts/video_review.py prepare --manifest data/regression/manifest.yaml --split-file data/regression/splits.json --output data/review/session-01.bundle.json
```

制作过程读取完整视频，保存源文件 SHA256、尺寸、原始帧索引以及每帧精确 PTS/time base。
Argus 与 CVAT 使用从 0 开始的帧号；FiftyOne 使用从 1 开始的帧号。适配器按这份映射转换，
不会用 `frame / fps` 猜测可变帧率视频的时间。原始缺失/无效时间戳会导致准备失败。
时间范围的端点为包含两端的原始帧号；区域坐标为原视频像素坐标。

复核包保存原始 PTS 的绝对时间，查看相对视频起点的时间时减去首帧 PTS 时间。
已有 pipeline 报告中的 `timestamp_sec` 仍可能来自 `frame_index / fps`；FiftyOne 将其作为
报告的旧时间字段保存，并与源视频的精确时间分开。对于可变帧率视频，不应把这个旧字段
当作精确 PTS，也不能直接沿用旧报告的事件延迟作为最终发布证据。复核用于修正标签和定位
错误，事件级指标仍需完整生产链路重新验收，CLI 不提供发布许可。

要附带预测，使用完整检测链生成的 `argus.pipeline-video-eval.v1` 报告：

```powershell
python scripts/video_review.py prepare --manifest data/regression/manifest.yaml --evaluation-dir data/regression/reports/candidate/video_eval --output data/review/session-01.predictions.bundle.json
```

`--evaluation-dir` 必须为每个 clip 提供 `<id>.json`，或在其 `video_eval/` 子目录提供同名文件。
缺报告、视频摘要或相机不匹配会失败。detector-only 报告不能被当作完整链路预测导入。
建议首次从无预测包开始人工标注；附带预测时注意避免让模型建议代替人工判断。

## CVAT 上传与回导

凭据仅通过 `ARGUS_CVAT_TOKEN`，或同时通过 `ARGUS_CVAT_USERNAME` /
`ARGUS_CVAT_PASSWORD` 读取；两种方式不能混用。不接收 `--password` / `--token`，
URL 也不能带账号、密码、query 或 fragment。bundle 与 receipt 不保存凭据。

PowerShell 的用户名/密码方式可以用安全交互提示，避免把密码写入命令历史：

```powershell
$cvatCredential = Get-Credential
$env:ARGUS_CVAT_USERNAME = $cvatCredential.UserName
$env:ARGUS_CVAT_PASSWORD = $cvatCredential.GetNetworkCredential().Password
python scripts/video_review.py cvat-push --bundle data/review/session-01.bundle.json --url https://cvat.example --receipt data/review/session-01.cvat-receipt.json
```

需要组织时，上传和回导都加 `--organization <slug>`。一次上传为每段视频创建可追踪任务，
receipt 记录任务 ID 与阶段。**保留 receipt 并复用相同路径重试，不要删除它再上传。**
若收到请求结果未知的错误，先在 CVAT 检查任务并依据 receipt 恢复；无法确定任务 ID 时
需要人工核对，工具会拒绝盲目重新创建，避免重复上传。

当前 receipt 格式为 `argus.cvat-receipt.v2`，同时绑定源预标注摘要、标签类型和属性语义。
旧 v1 receipt 不会自动升级或触发重新上传；升级前应保留原 bundle、receipt 和远端任务，
先核对已有任务再制定迁移，不能删除 receipt 后重推来绕过校验。

在 CVAT 中逐项核对事件区间、忽略区间和目标框，完成任务的人工复核状态。
事件区间上传为逐帧 tags，并保留独立事件 ID；目标框是另一维度。预置源框通过
`argus_source_geometry` 属性声明从属于原事件标签，回导时避免重复计数；人工新增的独立
目标框或轨迹保留自己的事件身份，即使与已有同类别事件重叠也不会被吞掉。
复制预置框来标新目标时，必须分配新事件 ID 并清除源几何标记；不同几何对象共享源事件 ID
会被拒绝。同一对象的连续标注应保留在原 track 内，不能靠克隆源属性绕过事件身份校验。
**删除框不会删除已有 tag 事件**；拒绝一个事件或更改类别时，需要同时检查并编辑 tags
和几何标注。只想用轨迹定义事件时，先去掉该事件原有 tags。
相邻或重叠但 ID 不同的事件会分别保留。忽略区间统一使用 `ignore` tag，其原因在
`argus_ignore_reason` 属性内；忽略是整帧语义，不支持使用区域框表示忽略。
回导必须任务完成并明确提供复核人。全部为空的片段还必须逐条确认正常；
仅仅没有目标框，不能证明整段视频已经检查完毕。

```powershell
python scripts/video_review.py cvat-pull --bundle data/review/session-01.bundle.json --url https://cvat.example --receipt data/review/session-01.cvat-receipt.json --reviewed-by operator-42 --confirm-normal-clip normal-01 --output data/review/session-01.reviewed.bundle.json
Remove-Item Env:ARGUS_CVAT_PASSWORD
Remove-Item Env:ARGUS_CVAT_USERNAME
```

`--confirm-normal-clip` 可重复。误写 clip ID、未完成任务或没有确认的全空标注均不应回导成功。
`reviewed_by` 是操作员的人工声明，不是认证服务签名；请把真实复核人员身份纳入现场流程。
不要把未检查的片段批量声明为 normal。CVAT 的 source/video/time mapping 如有改变应重新制作任务。
回导前后都会检查任务完成状态与更新时间；复核期间发生编辑会拒绝本次回导，应完成编辑后重试。
本轮支持逐帧 tags、轴对齐矩形 shapes/tracks；非空 `intervals`、旋转框、多边形、骨架和未知
标注集合会明确拒绝，不能清空或确认 normal 来跳过不支持的标注。

## 导出新的回归标签

```powershell
python scripts/video_review.py export-labels --bundle data/review/session-01.reviewed.bundle.json --split test --output-dir data/review/session-01-regression
```

所有导出的片段都必须已复核。已分配 split 的包必须明确选择 `--split`，只导出该集合；
不要把训练或调参片段混进最终测试 manifest。未分配的包可以不传 `--split`，
但不应把这种导出作为独立测试集的证明。命令生成新的 `<clip>.labels.json` 与 `manifest.json`，
manifest 引用原视频并使用新标签，保留相机、采集批次、参考图及预期检测类型。
JSON manifest 可以直接由现有 YAML loader 读取。原始视频、原 manifest 和原标签不会被修改。
输出默认不覆盖。标签保存于不可变的 `annotations/<generation-sha256>/` 目录，全部写完并
核对后才原子更新 `manifest.json`；`--overwrite` 只切换 manifest 到完整新一代标签。
中途失败会保留旧 manifest 及其标签，可能留下未被引用的新一代文件，不能把它们当作已发布结果。
CVAT receipt 没有 `--overwrite` 开关。

现有回归器评估的是“异常时段召回”，不是每个物体的召回。导出的 `positive_ranges` 会将
重叠及相邻范围合为不重叠时段，避免同一时段因多个框重复计数；完整对象事件仍保存在
`review_positive_ranges` 和 `boxes`，时段通过 `source_event_ids` / `source_labels` 保留来源映射。
标签带 `argus.review-regression-labels.v1` 格式标记，重新 prepare 时恢复对象事件而非重复导入时段。
新标签同时绑定视频摘要和 clip ID，不能挪给其他片段继续沿用 reviewed 状态。

如果原先标为 `negative` / `normal` 的片段实际存在正例，导出会拒绝类别冲突。
应在复核包中明确修正该片段的 `kind` 并复核，再重新导出；工具不会猜测它属于静态异物还是快速物体。

在测试机上使用经过批准的 profile 与模型运行完整检测链：

```powershell
python scripts/run_regression_suite.py --manifest data/review/session-01-regression/manifest.json --evaluation-mode pipeline --config configs/default.yaml --model data/models/candidate/model.pt --model-version-id candidate-version --output-dir data/regression/reports/candidate-reviewed
```

这条命令用于功能与质量回归。生产发布还需要原有严格解码、profile/模型绑定、完整语料、
覆盖率、FP/h、事件召回、P95 延迟及可信 soak 证据，复核包不能绕过这些门禁。

回归完成后，用新标签 manifest 和这次报告制作一个同时含真值和同源预测的包：

```powershell
python scripts/video_review.py prepare --manifest data/review/session-01-regression/manifest.json --evaluation-dir data/regression/reports/candidate-reviewed/video_eval --output data/review/session-01.reviewed-predictions.bundle.json
```

这个顺序保证 FP/FN 对照使用人工确认后的标签。最初附带建议的 predictions 包可能尚未复核，
不能据此把未标注帧当成正常，或把模型建议作为真实标签计算 FP/FN。

## FiftyOne 查看

```powershell
python scripts/video_review.py fiftyone-import --bundle data/review/session-01.reviewed.bundle.json --dataset-name argus-session-01
python scripts/video_review.py fiftyone-import --bundle data/review/session-01.reviewed-predictions.bundle.json --dataset-name argus-session-01-reviewed-predictions --launch
```

默认只创建持久数据集，不打开浏览器。`--launch` 才启动 FiftyOne App，默认载入
`argus-current` 已保存视图并保持进程等待，只展示当前完整导入批次；
退出 App 后结束。同名数据集只有在属于该 Argus 集成、suite 相同且源媒体身份不变时才允许
幂等更新；不属于该集成或存在身份冲突的同名数据集会被拒绝。要保留独立复核/预测快照，
使用新的数据集名字。
样本保留视频摘要、相机、profile、模型版本、采集批次和 split，帧保留原始索引与精确时间。
人工标签、模型预测、阻断原因分别展示；阻断检测的帧不会显示成正常预测。
报告中的旧时间单独保留为 `argus_eval_timestamp_sec`，不要把它与精确源视频时间混用。
精确绝对时间位于 `argus_timestamp_sec`，相对首帧时间位于 `argus_relative_timestamp_sec`。
只有人工复核且不在忽略区间内的帧才形成真值；未复核框和忽略区间内的框保留为建议，
FP/FN 根据这次人工真值和可用检测结果重新推导，报告原始 decision 仅作诊断保留。
历史样本会保留，排错视图仅针对当前完整导入批次。已有采集批次或视频的 split 身份
不能通过只重新导入部分片段来改变，避免历史训练素材进入测试集合。
FiftyOne 用于查错与筛选，Argus 的回归报告仍负责最终发布指标。

每个数据集固定一个导入主机，同机并发由文件锁串行化；跨主机更新同一数据集会被拒绝，
迁移主机时先使用新的数据集名字。长视频按 256 帧分批保存，失败批次保留为 pending，
可以在原主机用同一 bundle 重试。模型实际身份只取报告的 `model_identity.actual`，
请求版本和报告声明版本另存；缺少实际身份会显示 unavailable，不用请求值补成实际加载成功。

导入出错时不使用该数据集作验收依据。通常会隔离本轮样本并保存 failed 状态；若 MongoDB
完全不可写，连补偿或失败标记也可能写不进去。数据库恢复后应重新导入，并检查数据集
状态、当前 generation 和排错视图是否一致，再继续复核。这里不提供跨 MongoDB 文档的原子事务承诺。

当前 CLI 不把 FiftyOne 中的手工修改直接回写为验收真值；人工回导统一通过上述 CVAT 流程。
使用真实复核包加上同源 pipeline 预测，可避免比较不同视频或不同模型的结果。

### SceneGuard / LightGlue 分歧复核

`fiftyone-import` 可重复指定 `--scene-comparison <report.json>`，把离线 SceneGuard /
LightGlue 报告附到相机和源视频摘要一致的 clip。先按前文 `prepare` 制作原视频复核包，
再按 [LightGlue 旁路手册](scene-guard-lightglue-shadow.md) 生成报告，最后导入：

```powershell
python scripts/video_review.py fiftyone-import --bundle data/review/scene-review.bundle.json --dataset-name argus-scene-review-01 --scene-comparison data/reports/scene-guard/original-view.json --launch
```

有本选项时 `--launch` 默认打开 `argus-scene-disagreements`，无报告仍为 `argus-current`。
可切换 `argus-scene-raw-disagreements`、`argus-scene-errors`、`argus-scene-unreviewed`、
`argus-scene-reviewed` 和 `argus-scene-stale-review`。
报告映射按原始从 0 开始的帧号和精确 PTS/time base 校验，再转为 FiftyOne 从 1 开始的
frame number；记录绑定不用抽样序号或平均 FPS。未采样帧不计为两路一致。
精确复制 PTS/time-base 使用字符串字段 `argus_pts_exact` / `argus_time_base_exact`，
不要依赖浏览器对超出安全整数范围的旧数字字段的显示。秒数仍是浮点辅助信息。
FiftyOne 原生播放器的时间/帧号定位使用单一帧率，并不读取这些自定义 PTS 字段；因此，
带报告导入另做所有原帧的 nominal-FPS 网格预检，不满足时在 SDK/数据库操作前拒绝。
允许半个 time-base tick（至少 1ns）的量化误差，但不超过 `0.005 / fps` 秒；明显 VFR、
漂移或粗时间基可能被保守拒绝，不自动转码或改写源。具体边界及官方实现链接见
[旁路手册](scene-guard-lightglue-shadow.md#将分歧帧送入-fiftyone-人工复核)。
元数据预检不是浏览器画面已对齐的证明；不支持的源不能靠修改 fps 或复用旧报告绕过。
报告 `partial/error` 和限制信息单独保留，执行失败不是 normal 或姿态匹配成功。
报告没有完整 baseline 解码证据、零比较帧、源变化、重复报告归属或时间映射错误时拒绝导入。

上述原生视频导入阶段不再运行模型或解码视频，不需要 LightGlue、Torch/CUDA、权重和原 profile 目录；
但源视频仍须可读以核对摘要。这条诊断链与已有 pipeline prediction / ground-truth 字段
隔离，不会改写它们。bundle 显式声明的 `scene_profile_id` / `model_version_id` 必须与报告的
profile / 声明模型版本一致；未声明时只记录报告身份，不推断模型授权。旧参考图路径的图像
摘要不作为新报告参考图的强制匹配条件，二者仍分别保留来源。
人工字段 `argus_scene_human_state`、`argus_scene_human_note` 和
`argus_scene_reviewed_report_sha256` 仅用于复核笔记；合法状态为 `same_pose`、`moved`、
`uncertain`、`unreviewable`，且复核摘要匹配 `argus_scene_report_sha256` 才算当前报告已查看。
换报告不会把旧确认自动沿用；单独看到保留的 human_state 文本不表示当前确认有效。
使用 reviewed / unreviewed / stale-review 动态视图区分；恢复逐字相同的旧报告仍认可原摘要确认，
仅移动报告路径也不失效。导入器不赋值人工字段，避免其旧空值覆盖并发保存的笔记。
导入帧 schema 升级会重写算法字段，但保留人工记录；这仍不是导入与人工编辑之间的事务。
App 任意标量字段编辑能力尚待验证，Python API 保存示例见
旁路手册。这里不生成 pose 真值或发布证据，不会改变现有 SceneGuard 放行和恢复逻辑。
报告文件 SHA256 只标识报告，不是可信签名；`source_verified_against_bundle` 是与 bundle
相机、媒体摘要和 PTS 映射的对照。importer 会再次哈希实际源视频，但因不重新解码，
逐帧 tensor SHA256 只是报告声明的显示值，不是导入器独立验证的图像身份。

### VFR 源的受控逐帧图片复核

对于原生视频模式因 PTS/FPS 网格不满足而拒绝的源，使用独立图片模式，不修改 fps、
不转换或覆盖源视频，也不复用原生视频数据集的名字。下面命令仅供恢复后的复核环境执行，
本轮没有在本机运行命令、解码、SDK、服务或测试。

图片准备使用严格 FFmpeg/ffprobe 顺序解码并检查完整 EOF，按报告绑定的**原视频帧号**
取图，逐帧对照精确 PTS/time base 和 BGR tensor SHA256。不会使用平均 FPS seek、
抽样序号或播放器截图替代原始身份；不重跑 SceneGuard/LightGlue 或加载模型。
选中帧写为原尺寸无损 PNG，新的 manifest 绑定 PNG 文件摘要、原始帧摘要、源视频摘要、
精确时间和报告 SHA256，并保留报告状态、抽样/截断限制及诊断摘要。

准备图片的专用复核环境还需安装 NumPy，并在 PATH 提供支持该源解码的
`ffmpeg` / `ffprobe`。这不是安装 LightGlue extra 或生产 GPU 推理环境；不要把它加入
生产启动流程。普通图片导入不重新解码视频或运行模型，但仍要求原始视频、bundle、
原始旁路报告和生成的 PNG 可读，以重新核对来源及摘要。

```powershell
python scripts/video_review.py scene-images-prepare --bundle data/review/scene-review.bundle.json --scene-comparison data/reports/scene-guard/original-view.json --output-dir data/review/scene-images-01 --selection disagreements --max-images 10000 --max-output-bytes 10737418240
python scripts/video_review.py scene-images-import --bundle data/review/scene-review.bundle.json --scene-comparison data/reports/scene-guard/original-view.json --manifest data/review/scene-images-01/manifest.json --dataset-name argus-scene-images-01 --launch
```

两个命令均必须显式提供 `--scene-comparison`，可重复为不同 clip 指定报告；只处理已绑定报告
的比较集合。默认 `--selection disagreements` 包含最终状态分歧、raw-only 分歧和算法错误帧；
需要全部已比较帧时使用 `--selection all-compared`，未采样帧仍不补成正常或已复核。
`--max-images` 默认且最多 10000，`--max-output-bytes` 默认且最多 10 GiB；可显式降低，必须是正整数。
超限明确失败，不静默截断，也不降低分辨率；完整解码验证不因只选择少量帧而省略尾部。
无分歧或无可选图片不表示算法通过，仍须核对报告错误与覆盖范围。

`--output-dir` 必须是新的专用目录，没有 `--overwrite`。不能指向源输入或受保护 profile
目录；校验、解码、摘要或预算失败时不得发布一个伪装为完成的 manifest。
不要手工更改 PNG、manifest 的帧号/PTS/报告摘要来绕过拒绝；导入前会重新核对。
PNG 文件摘要只能证明保存文件身份，报告摘要只是来源追踪，不是可信签名或生产发布授权。

图片模式一张图片对应一个 FiftyOne **sample**，不使用视频 `sample.frames`，也不经过
原生视频播放器的单 FPS 定位。`--launch` 默认打开 `argus-scene-unreviewed`，其余分歧、
raw 分歧、错误、reviewed、stale-review 和当前批次视图按 sample 筛选。
原始从 0 开始的 `argus_frame_index`、从 1 开始的 `argus_frame_number` 和精确
`argus_pts_exact` / `argus_time_base_exact` 作为来源元数据保留；不要把图片在列表中的位置
当成视频帧号。图片查看不能代替原视频时序上下文和真实浏览器验收。

FiftyOne 导入没有全局事务，导入和失败恢复期间保存视图可能短暂只显示子集；必须同时
核对 `dataset.info["argus_scene_images"]["state"] == "complete"`。`importing/failed`、
空视图或部分视图均不能当作整批成功；数据库不可写时，pending 隔离补偿也可能失败。
上述故障恢复仍须在真实复核环境验收，本轮没有验证其运行结果。
每次图片导入或失败重试都必须使用**新的独立 CLI 进程**，重导前先在人工复核会话保存笔记。
导入器拒绝已加载过 FiftyOne 或已尝试图片导入的 Python 进程，不允许在笔记会话原地重试。
SDK 首次加载前关闭单例缓存和数据库管理权限，并预读 FiftyOne 的实际 JSON 配置，拒绝含
非空或非法 `module_path` 初始化钩子的配置。空环境变量不能覆盖配置文件里的钩子，不能只
清空环境变量后继续导入。使用无钩子的专用配置，导入期间不要并发修改配置；不复制配置或打印连接凭据。
SDK 加载后再次校验实际配置；缺少支持时失败关闭。
不再对已有对象执行硬刷新，不替人工保存或丢弃其未保存草稿；其他会话的草稿仍须由本人保存。
数据库、数据集及锁 collection 必须使用 `PRIMARY`、显式 `readConcernLevel=local` 和有确认的写入；
未显式指定读取级别（包括客户端返回 `None`）、`majority/snapshot`、从节点读取、`w=0` 或无法
核实这些能力的连接均在目标预检前拒绝。即使读主节点，`majority` 视图也可能暂未包含上个
`w=1` 导入已确认的登记；未指定级别又可能继承服务端默认值。请在专用数据库原 URI 中明确
设置 `readConcernLevel=local`，保留原数据库、认证、TLS 和其他配置，修正后开新 CLI。
不要只给一次原始查询改读取参数，却让 SDK 的样本、视图或元数据仍使用旧的读取策略。

报告更新后须用**新目录**重新准备图片，再导入同一个独立图片数据集；导入不会修改人工
笔记或替用户确认新报告。人工确认摘要仍指向旧报告时，在 `argus-scene-stale-review`
中可见，且仍属未复核；只有人工核对当前报告后，设置合法人工状态并把
`argus_scene_reviewed_report_sha256` 明确设为当前 `argus_scene_report_sha256`，才算当前报告
已查看。报告变更不代表同一原始帧变更；只有源媒体和帧身份一致时才允许保留其人工记录。
原生视频数据集与图片数据集的所有权不同，不允许相互复用。
新报告未再选中的旧图片仍保留为历史样本，不出现在当前批次保存视图；集合中消失不表示
新报告已复核。需要当前全部比较帧时，准备时显式使用 `--selection all-compared`。
同一 clip 首次补齐采集批次或从 `unassigned` 分配 split 时，允许历史未选中图片保留原来的
未分配元数据；重导会按同一原始媒体身份归一化检查，避免局部刷新后无法重试。
已分配 split 或非空采集批次仍不可改绑，同一视频/采集批次跨 clip 的 split 隔离不放宽。
历史元数据补齐不替代新报告的人审摘要确认，也不会改写人工备注。
源身份约束还会存入带版本的数据集登记：即使 clip 本次没有选中图片或以后未出现在 bundle 中，
它仍参与身份与 split 隔离检查。新数据集第一次保存所有权时就保存完整源登记，后续登记更新
在样本更新前随 `importing` 状态保存；失败重试不能解除已提交的绑定。
完整登记使用 `argus.scene-review-image-sources.v2`。缺少登记或仍为旧 v1 的既有数据集均拒绝
自动迁移：历史图片无法证明零图、无报告或已缺席来源的完整身份，而 v1 也未记录其是否
来自不完整的自动迁移。旧数据和人工备注保持不变，请使用新数据集；若必须延续旧集合，
须另行取得可信的完整历史来源凭据并制定显式迁移方案，不能改 schema 字符串、手工补空登记
或自动复制“已复核”确认来绕过拒绝。
登记损坏、缺失或超出体积预算同样拒绝导入。任何历史图片所属 clip 不在持久 v2 登记中也拒绝，
不能从图片反推补齐缺失来源；当前 bundle 中尚无历史图片的新 clip 仍可正常登记。
首次所有权登记在提交前失败时，可能留下没有可信 owner 的空集合；导入器不会自动接管或删除，
应换新名字创建。若登记已提交但确认丢失，在新 CLI 进程重试会重新读取并保留已落库的约束，
不按上次失败进程中的脏缓存解绑或补提交。
导入还会获取专用数据库集合 `argus_scene_image_import_locks` 中按数据集名唯一的互斥记录，
覆盖不同用户或不同临时目录的 CLI；进程内的临时文件锁不再是唯一串行化保障。
启用前须停止旧版本导入器；不能混跑不参与数据库互斥的旧客户端，或在导入期间直接改写受管登记。
插锁前还须证明锁存储是普通、非 capped、非时序、非 view/clustered、无 TTL 的 collection，
且有普通默认 `_id_` 唯一索引。集合及所有索引（含 `_id_`）只允许缺省或 `{locale: "simple"}` 的二进制
排序规则；其他 collation（包括 numericOrdering）可能令 token 比较不再逐字精确，直接拒绝。
元数据或索引不可读、结构不兼容时失败关闭，不自动删除、
重建或清空既有集合；锁集合不存在时才显式创建普通集合，并重新核对结构。部署仅支持独立 mongod 或
普通副本集，不支持 mongos，也禁止直连分片节点；导入期间不得并发 drop/recreate 集合或修改索引。
互斥记录在读取目标数据集前获取，正常完成或可确认的本地失败才按自身随机 token 释放。
持久层／传输异常、超时、取消或中断不能证明服务端写入已经停止：此时停止后续补偿写并保留锁，
包括被补偿逻辑捕获的未知写失败；连错误分类本身失败时也保守保锁，不自动接管或按时间清理。
锁释放失败会同时报告主导入失败与清理失败的类型；未知 SDK 细节不直接写入诊断摘要，
CLI 另遮蔽 MongoDB URI；SDK 错误文本无法安全转换时仅显示类型及固定占位，不再触发二次诊断崩溃。
即使元数据已显示 `complete`，锁或写入结果不确定仍按 CLI 失败处理。
此互斥不是所有样本/视图的全局事务，也不替代数据库持久性保障；副本集部署应配置多数确认写入
并验收故障切换，防止数据库自身回滚已确认的登记或锁。
通过只读元数据查询先检查持久 owner/suite/host、媒体类型及 v2 登记，再加载已有数据集；
加载后重新核对持久身份、登记和新对象。旧 v1、无可信 owner 或不匹配目标在数据集加载前拒绝。
目标数据集版本必须与当前 SDK 完全一致；图片导入不承担 SDK/数据库迁移，请由管理员另行
完成兼容性检查和备份后的显式迁移。请使用专用复核数据库；SDK 初始化仍可能写自身数据库
配置，退出时也可能清理非持久数据集，不是零写沙箱，不应连接无关业务数据库。

### 图片导入遗留锁的人工恢复

以下是测试环境恢复后的管理员流程，本轮未连接数据库或执行清锁。不要因 PID 消失、时间久、
客户端超时或 `killOp` 返回成功就认定可以解锁；服务端操作可能尚未到达中断点或仍将迟到提交。

1. 暂停新的导入入口，停止所有相关导入客户端，保留锁与故障信息；确认人审笔记的实际保存状态，
   不把未保存草稿当成已落库记录，也不通过改写人审字段、owner 或 source registry 来“修复”锁。
2. 管理员确认相关服务端读写操作已经结束，而不只是客户端退出或终止请求已被接收。
   无法确认操作静止时保持锁定，不发补偿写、重导或删除请求。
3. 在专用复核数据库以 `PRIMARY` 和显式 `local` 重新读取锁，并确认集合/索引使用 simple 二进制
   比较，核对数据集名、主机、PID、创建时间和**当前原 token**。保留记录副本；仅允许同时匹配
   `_id + 该 token`、显式 simple 比较的条件删除，禁止仅按数据集名、主机或时间清理。
4. 删除必须确认成功且恰好删除一条。若响应丢失或异常，继续暂停入口，先重新读取并核查服务端；
   原 token 仍存在、读取不明或出现新 token 时都不能盲目重试删除，更不能删掉后来者的锁。
5. 确认锁结果及持久登记后，以新的独立 CLI 重试完整校验。首次登记未提交的无 owner 孤儿仍不接管，
   已提交的来源约束和人审备注继续保留；重试成功前不把残留样本或保存视图当成整批成功。

图片复核同样不回写 CVAT 或 Argus ground-truth，不批准 profile、模型或恢复评分/告警。
生产 SceneGuard 和原生视频模式的 PTS/FPS 拒绝逻辑均保持不变。Python API 人工记录示例见
[旁路手册](scene-guard-lightglue-shadow.md#vfr-受控逐帧图片复核)。

## Dashboard 场景、基线、训练与发布工作流

模型页的基线入口新增一条受控流程，与离线 CLI 共用
`argus.validation.scene_profile_baseline.build_scene_profile_baseline`；不是在 HTTP 路由中导入脚本。
新 API 位于 `/api/scene-workflow/{camera_id}`，使用统一 envelope、严格请求 DTO 和既有 RBAC。
开发阶段先完成源码和静态检查；2026-10-22 获准在本机运行普通功能验证，范围与结果见本节末尾。
真实训练、模型识别及 GPU 验收仍不在本机执行。

1. 工程人员从本相机的既有基线图片中显式选择参考，并核对服务端 SHA256、确认它是目标正常
   场景，创建新的 workflow/profile ID。只冻结一个 `validation_only` 草稿，不采用实时首帧、
   不自动批准或激活。参考内容变化会要求重新确认；既有 ID/产物不可覆盖。
2. 在专用数据根的 `regression/` 中准备至少两个不同 `source_session_id` 的正常录制及完整
   `argus.normal-video-review.v1` 审核 JSON。填写相对视频/审核路径，不能提供 URL、绝对路径、
   `..`、符号链接或 reparse 路径。审核身份、时区时间、源 SHA、正样本和忽略区间继续由共用
   构建器严格校验；session 的声明不能代替真实独立录制。
3. 提交 strict v3 基线任务。Web 固定 `strict_release_decoder=true`、`allow_alignable=false`，
   最少训练 100 张、校准 50 张及既有时间跨度门槛；不暴露降低门槛或跳过质量校验的参数。
   输出是 `baselines/<camera>/default/scene-<workflow>/`，包括原始 PTS/EOF 证据和清单。
   `training_ready` 不等于 `release_eligible`，也不代表识别质量通过。
4. 从该工作流提交训练，只生成 `pending_confirmation` 作业；到现有训练页人工确认。
   提交时、执行时和登记前重验完整基线树、manifest/reference 摘要及逐 session train/calibration
   角色，不允许随机拆相邻帧、混合旧 dataset selection 或跳过质量检查。实际运行 artifact
   必须完整覆盖冻结的独立校准集合；读图或推理遗漏不能用剩余 50 张冒充成功。校准非 `ok`
   时不能登记为该流程的候选模型。登记和 profile 审批还会读取真实校准 JSON，复核冻结样本数、
   分数数量/有限性/顺序及阈值结构，不接受仅有 `ok` 标记的空壳。创建、确认、拒绝训练及场景
   审批的操作人以认证会话为准，JSON 不能冒名；空白或 unknown 身份不能进行这些写操作。
5. 训练完成后另行批准 profile/model 绑定。服务端从该工作流唯一训练作业读取 registry 模型，
   复核 camera、job、baseline、校准和完整 runtime artifact；保存真实 `approved_by/approved_at`。
   重试只能核对同一个完整绑定，不能接管同名其他 profile，也不能覆盖原审批人/时间。
6. 批准后可显式“选择活动场景（不切模型）”。它按相机互斥、分别原子更新 baseline current
   与 profile active 指针，不修改运行模型或 applied stage。请求必须携带确认时的活动 profile ID
   和 `active.json` 内容摘要；服务端锁内比较，迟到请求及 A→B→A 后的旧确认均返回 409，须
   刷新后重新确认，不能用当前值自动重试。两文件并非全局事务，部分失败会
   标记 `selection_incomplete` 并阻断后续操作，管理员须核对两个指针，不能当成已恢复检测。
   旧运行模型与新场景不匹配时，后续重载/重启仍由 SceneGuard 拒绝。
7. 在既有模型页走唯一 `ModelReleaseService` 的 shadow/canary/production 流程。独立完整
   回归、当前全部指标、人工授权和可信 72 小时 soak 缺任一项，都保持 production NO-GO。
   本流程不提供自动发布、不改变 SceneGuard 判定、不回写源视频，也不把人审报告当发布授权。

工作流记录在同一数据根的 `scene_workflows/<camera>/<workflow>/workflow.json`。文件锁不自动过期
或偷取；发生遗留锁、任务提交结果未知或部分场景选择失败时，须确认后台操作静止并人工核对。
不要删除基线或改摘要来“修复”失败。训练提交先冻结 job ID，数据库结果未知时不自动重复创建。
任务并发额度或同相机同类占用在预留前明确拒绝时，基线工作流回到可重试失败态；线程启动或
提交结果未知时仍保留预留。队列任务由数据库条件 UPDATE 原子领取，同一作业仅一个执行者。
运行超过时间上限只标记需核查，不能据此认定 worker 已停止并自动转为 failed；管理员须先
确认相关进程/线程及迟到写入均已停止，再处理记录。完成/失败写入不能覆盖已变更的终态。
SSL backbone 的登记、活动标记与作业完成同事务提交，失败回滚不改变原活动 backbone。
刷新失败保留上次快照并标记过期、禁用写操作；WebSocket 事件和 REST 对账共同更新状态。

旧实时采集/旧随机拆分训练入口仍为兼容诊断路径，不因页面保留就具备 production 资格。
当前严格发布门尚无仅静态能力分档，详见 [正式发布契约](../acceptance.md#当前-production-发布契约沿用严格门禁不作能力分档)。

测试机恢复后除原测试外，须验收该完整流程，注入构建失败、提交前后故障、场景/源/基线摘要
变化、跨 workflow/job 误绑定、校准跳过/失败、未授权请求、冒名审批、并发选择及两指针部分失败。
分别核对候选、所选场景、运行时 applied identity 和最终 release-readiness，禁止相互替代。

以下是完整工作流的 targeted tests 清单；含真实训练/媒体处理的项目仍须在测试环境执行。
本机不要直接运行该完整清单，而使用下方有阻断保护的普通功能入口：

```powershell
python -m pytest tests/unit/test_scene_baseline_service.py tests/unit/test_scene_training.py tests/unit/test_scene_training_execution.py tests/unit/test_training_artifacts.py tests/unit/test_scene_workflow.py tests/unit/test_scene_workflow_routes.py tests/unit/test_scene_training_route_authority.py tests/unit/test_scene_workflow_task_concurrency.py tests/unit/test_training_job_claims.py tests/unit/test_scene_baseline_runtime_wiring.py tests/unit/test_model_deletion_events.py -v
```

场景工作流新增/收紧了 OpenAPI DTO；本机普通验证阶段已重新生成契约，并通过类型一致性门禁。
这只验证 API/页面契约，不授权模型训练、推理或正式发布。

### 本机普通功能验证（禁训练和识别）

用户于 2026-10-22 允许本机执行普通测试。固定使用仓库 `.venv`，不要运行全量 pytest 或启动
`python -m argus`。受控入口按显式清单选取普通 API、权限、数据库、任务并发、场景工作流、
发布元数据和复核契约；真实训练器、检测器、pipeline 初始化及模型 SDK 被阻断，只允许模拟运行时。
真实相机/视频管线启动、模型加载、模型质量、CUDA 和真实 72 小时 soak 均不在这次验收范围。
CVAT/FiftyOne 用例使用模拟 SDK/数据库，不能视为真实外部服务联通验收。

```powershell
.\.venv\Scripts\python.exe scripts/run_local_functional_tests.py --suite core
.\.venv\Scripts\python.exe scripts/run_local_functional_tests.py --suite dashboard
.\.venv\Scripts\python.exe scripts/run_local_functional_tests.py --suite review
.\.venv\Scripts\python.exe scripts/run_local_functional_tests.py --suite lint
.\.venv\Scripts\python.exe scripts/run_local_frontend_tests.py unit
.\.venv\Scripts\python.exe scripts/run_local_frontend_tests.py build
```

Windows 沙箱若卡在 `asyncio → socketpair → accept`，需允许本机回环通信后重新执行受控入口；
不要因此修改 API 逻辑或放开模型阻断。前端临时缓存固定在工作区 `.local-validation/`，避免系统
沙箱临时目录重命名失败。该目录已忽略，不提交数据库、截图、trace 或缓存。

浏览器只连接独立测试服务，不能使用真实运行数据库/配置。第一个终端运行：

```powershell
.\.venv\Scripts\python.exe scripts/run_local_functional_server.py
```

第二个终端运行：

```powershell
.\.venv\Scripts\python.exe scripts/run_local_frontend_tests.py e2e
```

服务只绑定 `127.0.0.1:18080`，在 `.local-validation/dashboard-*/` 新建临时配置、数据库与测试
账户；不启动 go2rtc、调度器、训练器或摄像头。E2E 先验证专用隔离身份标记，不匹配即拒绝测试写入。
`local-admin/local-viewer` 和 `Argus-local-only!42` 仅属于这些临时数据库，禁止复用到真实部署。
测试结束关闭该服务即可；不得按端口或进程名误杀其他服务。

已通过的后端套件：core 1178、dashboard 242、review 717、lint 71（共 2208）；前端 244 项单元
测试、TypeScript、正式构建及 Ruff 全仓检查通过。浏览器最终确认轮 52 项全部通过，覆盖四种
视口，包含全部非演示管理子页、登录/退出、RBAC、401/403/404/409/500、双会话状态及 WS 断线恢复。
前轮曾有一项 Chromium `ERR_INSUFFICIENT_RESOURCES`，其单独复验及随后完整确认轮均通过；
原失败 trace 与确认轮产物分别保留，未设置自动重试来掩盖失败。
本机实际环境为 Python 3.11.9、pytest 9.0.2、Vitest 4.1.10、Vite 8.1.4、Playwright 1.61.1。
可信 soak 单测使用模拟时钟、HTTP 和设备信息，并不代表执行过真实 72 小时长稳。
这些结果不表示 GPU 检测目标或 production readiness 已通过，既有发布门禁继续生效。

## 测试机恢复后的验收

1. 在独立复核环境保存依赖版本，确认 CVAT 2.79、SDK、FiftyOne 及 MongoDB 能正常工作。
2. 运行新集成的 targeted tests，检查帧 0/最后一帧、可变帧率、轨迹、空片段确认、
   断网上传恢复、重复上传、源文件变化、跨 split 泄漏、未知标注拒绝及拒绝覆盖。
3. 使用两段已知视频做 CVAT 上传、人工复核、回导，再验证新标签可被现有 regression runner 读取。
4. 打开 FiftyOne，逐帧对照原视频与标注坐标，并注入一段 SceneGuard blocked 结果检查显示语义。
   额外确认 FP/FN 保存视图确实包含已知错误帧；注入帧分批保存、视图保存和元数据保存失败，检查恢复。
5. 仅在 GPU 测试机跑完整 pipeline 回归。任何集成可用性结果都不等于生产模型质量已通过。
6. 附加已知分歧、raw-only 分歧、算法执行错误和抽样/截断报告，核对第 0 帧、最后一帧、
   负起始/大整数 PTS 及受支持 CFR 的浏览器实际 seek；不支持 VFR 应在 SDK 前拒绝。
   纯桥接对 VFR 保留精确映射，不等于 App 已对齐。检查无分歧空视图、missing/null 复核字段、
   人工摘要确认、并发笔记保存、同报告重导、A→B→移除→原样 A 与 schema 升级刷新。
   不提供 LightGlue 依赖/权重也应能导入已有报告；核对源变化、重复归属和不完整解码明确拒绝。
7. VFR 图片模式对照原始逐帧解码的首帧、最后一帧、稀疏比较帧、负起始/大整数 PTS，核对
   原尺寸 PNG、tensor 摘要、精确 PTS 和报告 SHA；检查不经过 FPS seek 或转码。
   注入损坏尾部、图片/报告/源视频变化、输出已存在、受保护目录、符号链接和预算超限，
   确认失败不会留下完成 manifest 或启动 App。用报告 A→B→原样 A 重导独立图片数据集，
    检查笔记保留、动态 stale/unreviewed/reviewed 语义和 sample 级视图；检查原生视频模式仍拒绝 VFR。
    覆盖全图导入后缩小集合并首次补齐 split/session、失败后重试及真实已分配身份冲突。
     另覆盖零选中/未附报告 clip 的身份持久化、缺席 clip 跨 split 检查、旧 v1 及不完整 v2 安全拒绝。
     注入登记提交前失败、提交后抛错、补偿也失败，以新 CLI 进程重试，核对持久约束和备注保留，
     确认没有加载旧缓存触发的隐式补提交或硬刷新。已加载 SDK 的笔记会话应在任何 SDK 调用前拒绝导入。
     额外覆盖配置文件中的初始化钩子在 SDK 前拒绝、从节点读和无确认写拒绝、不同临时目录的
     并发导入互斥、锁获取/释放失败与遗留锁人工恢复；不能因锁失败覆盖来源绑定或人审备注。
     额外检查既有时序/capped/TTL/无唯一索引/分片锁存储在插锁前拒绝、并发建普通集合的重查；
     模拟服务端迟到提交和未知补偿写，确认锁保留、新 CLI 无法进入，服务端静止且按原 token
     人工恢复后再重试。另覆盖非 simple/非法 collation、SDK 与锁句柄未显式 local 的预拒绝，
     错误分类器自身中断时保锁及不再补偿。主失败与释放失败必须同时可见且不泄露 URI 凭据，
     异常 `__str__` 抛错、返回非法类型或中断时 CLI 仍给出受控失败。

targeted tests 的待执行命令：

```powershell
python -m pytest tests/unit/test_video_review_contracts.py tests/unit/test_video_review_corpus.py tests/unit/test_cvat_integration.py tests/unit/test_scene_comparison_review.py tests/unit/test_fiftyone_integration.py tests/unit/test_scene_review_images.py tests/unit/test_fiftyone_scene_images.py tests/unit/test_video_review_cli.py tests/lint/test_lightglue_shadow_isolation.py tests/unit/test_scene_review_release_isolation.py -v
```

上游资料：[CVAT Python SDK](https://docs.cvat.ai/docs/api_sdk/sdk/)、
[CVAT SDK 包元信息](https://pypi.org/project/cvat-sdk/)、
[FiftyOne 安装](https://docs.voxel51.com/installation/index.html)、
[FiftyOne 视频标签](https://docs.voxel51.com/user_guide/using_datasets.html#video-datasets)、
[FiftyOne 已保存视图](https://docs.voxel51.com/user_guide/using_views.html)、
[FiftyOne 与 CVAT](https://docs.voxel51.com/integrations/cvat.html)。
缓存边界依据：[FiftyOne 配置源码](https://github.com/voxel51/fiftyone/blob/v1.8.0/fiftyone/core/config.py)、
[数据集加载与刷新](https://github.com/voxel51/fiftyone/blob/develop/fiftyone/core/dataset.py)、
[公开数据库连接 API](https://docs.voxel51.com/api/fiftyone.core.odm.database.html)、
[ETA 环境变量解析](https://github.com/voxel51/eta/blob/master/eta/core/config.py)、
[MongoDB 读写设置继承](https://www.mongodb.com/docs/languages/python/pymongo-driver/current/crud/configure/)、
[majority 读取视图与复制延迟](https://www.mongodb.com/docs/manual/reference/read-concern-majority/)、
[local 读取边界](https://www.mongodb.com/docs/manual/reference/read-concern-local/)、
[删除操作的排序规则继承](https://www.mongodb.com/docs/manual/reference/method/db.collection.deleteone/)、
[时序集合的 `_id` 语义](https://www.mongodb.com/docs/v8.0/core/timeseries/timeseries-bucketing/)、
[TTL 自动删除](https://www.mongodb.com/docs/manual/core/index-ttl/)、
[PyMongo 未知写入结果](https://pymongo.readthedocs.io/en/stable/api/pymongo/errors.html#pymongo.errors.NetworkTimeout)、
[MongoDB 操作终止边界](https://www.mongodb.com/docs/manual/reference/command/killop/)。
