# SceneGuard / LightGlue 离线旁路对比

本轮入口是独立命令 `scripts/compare_scene_guard.py`。同一段已录制视频、同一个已批准
profile 和参考图，分别通过现有 SceneGuard 与 ALIKED + LightGlue，输出逐帧状态、
几何证据、耗时、失败原因及分歧统计。命令不启动 Argus 服务、不连接摄像头，也不写
profile、模型注册表或检测状态。它不改变现有 `allow_alignable`、放行条件或恢复计数。

当前没有测试机。以下安装、命令及测试步骤均是测试环境恢复后的待验收事项，本轮未执行。
本机不要执行安装、CLI、模型加载、测试或构建。

## 运行边界

- 生产进程不导入 LightGlue；通过显式离线 CLI 才加载可选模型依赖。
- classic pass 先完整执行，再初始化 LightGlue 并进行第二次完整解码，避免上游 Torch
  全局设置影响 classic pass。两路针对相同原始帧号比较，不使用摄像头实时队列。
- classic 的比较范围是 `standalone_scene_guard_full_check`：独立 SceneGuard 完整检查。
  它不代表生产 pipeline 的 quick check、缓存、时序恢复或最终可用状态。
- 从配置中的 `profiles_dir` 只读加载指定相机/profile，必须已经 `operator_approved=true`、
  非 validation-only，并声明格式有效的模型版本和摘要绑定。本命令不加载生产异常模型，
  也不查询其运行时注册授权；报告中的绑定只表示 `declared`，不能作为发布许可。
- LightGlue 判断及分歧只写诊断报告；新方法返回 stable 也不会让生产恢复评分或告警。
  CUDA 不可用、权重不匹配、匹配异常、解码失败均应可见，不能计为匹配成功。
- 两个 pass 都需要读完整源视频。`--sample-every` 和 `--max-frames` 限制比较集合，
  不跳过尾部解码完整性检查。抽样或帧数限制必须在解读报告时保留，不能冒充全视频覆盖。
- 独立进程仍可能与生产共享 GPU/显存/磁盘；正式旁路调度前要现场测量资源竞争。

## 依赖与权重

可选 extra 名为 `lightglue`，固定官方仓库提交
[`eb42fee2d71449efb0aa5c10549752b5d75384d8`](https://github.com/cvg/LightGlue/commit/eb42fee2d71449efb0aa5c10549752b5d75384d8)，
没有将浮动 `main` 或 PyPI 同名包作为安装来源。默认安装不会安装该 extra。
固定提交用于可复核的源码适配，尚未完成本项目 CUDA 环境的运行认证。
适配器在导入模型库之前核对 PEP 610 安装来源：必须来自上述官方 Git 仓库和固定提交，
来源缺失、不符或实际导入路径被同名模块遮蔽会明确拒绝初始化。通过来源核对不代表安装目录未被后续修改，报告仍明确
记录 `installed_tree_integrity_verified=false`；验收时还要保存并复核实际环境。

恢复测试机后，在专用离线对比环境中安装与测试机驱动匹配的 PyTorch/Torchvision，
再从仓库根目录执行：

```powershell
python -m pip install -e ".[lightglue]"
```

该命令需要网络取得固定源码及其依赖；不能在生产运行时尝试联网补权重。保存最终依赖锁定
清单以及 Python、Torch、CUDA、OpenCV、PyAV 和官方源码提交。离线运行环境应预先准备
受信任的两个权重：

| 文件 | 对应官方来源 |
| --- | --- |
| `aliked-n16.pth` | [官方 ALIKED 模型](https://github.com/Shiaoming/ALIKED/tree/main/models) |
| `aliked_lightglue.pth` | [LightGlue v0.1_arxiv 发布](https://github.com/cvg/LightGlue/releases/tag/v0.1_arxiv) |

从可信来源取得文件后单独登记 SHA256，并把摘要与文件一起保存在测试记录中。
示例中 `Get-FileHash` 的结果只核对当前文件身份，不等同于上游真实性证明。
运行适配器强制要求两个本地路径和两个 SHA256，验证后以 `weights_only=True` 读取，
严格检查 state dict；不会自动下载，也不会以随机参数替代失败权重。

官方说明 LightGlue 代码及其预训练权重采用 Apache-2.0；ALIKED 采用 BSD-3-Clause。
本次固定使用 ALIKED-n16，未选择具有不同许可条款的 SuperPoint。
分发时保留对应上游许可和声明。[LightGlue 许可说明](https://github.com/cvg/LightGlue#license)、
[固定版本 ALIKED 文件声明](https://github.com/cvg/LightGlue/blob/eb42fee2d71449efb0aa5c10549752b5d75384d8/lightglue/aliked.py)。

## 待验收命令

以下示例只在恢复后的 CUDA 测试机运行，从仓库根目录调用。先确认配置中相机和已批准
profile 与视频来源对应；不要为了让命令通过而修改批准标记或绑定摘要。

```powershell
$extractorSha = (Get-FileHash data/weights/aliked-n16.pth -Algorithm SHA256).Hash.ToLowerInvariant()
$matcherSha = (Get-FileHash data/weights/aliked_lightglue.pth -Algorithm SHA256).Hash.ToLowerInvariant()
python scripts/compare_scene_guard.py --config configs/default.yaml --camera-id camera-01 --video data/regression/original-view.mp4 --profile-id approved-profile-id --extractor-weights data/weights/aliked-n16.pth --extractor-sha256 $extractorSha --matcher-weights data/weights/aliked_lightglue.pth --matcher-sha256 $matcherSha --device cuda:0 --output data/reports/scene-guard/original-view.json
```

默认 `--max-keypoints=2048`（允许 32–8192）、`--sample-every=1`，没有帧数上限。可以显式设置
`--sample-every 5 --max-frames 300` 做有限比较，但报告应保留限制，正式覆盖验收仍使用
完整比较集合。`--device cpu` 是显式诊断选择，CUDA 失败时不会自动切换到 CPU；
CPU 结果不能用来满足测试机 CUDA 验收。

报告文件默认禁止覆盖。确需替换同一份诊断报告时指定 `--overwrite`；即使显式覆盖，
也只允许替换已有、同 schema 且声明无放行权限的旁路 JSON 报告，不能覆盖其他普通文件。
禁止将输出指向源视频、配置和两个权重；所有已配置相机的 `profiles_dir` 整个目录树都受
保护，包括未选中相机的 profile、尚不存在的 `active.json` 和经目录别名解析后的路径。
输出目录应专用于诊断报告，不要将诊断命令放进生产服务启动脚本。

| 退出码 | 含义 |
| --- | --- |
| `0` | 报告 `status=complete`，无比较截断或执行错误；允许存在两方法分歧和已声明抽样 |
| `1` | 报告已写入，但为 `partial/error` 或未知状态，需查看错误和覆盖信息 |
| `2` | 参数、前置输入校验或报告写入异常，不能认为已有完整报告 |

通过前置校验后，缺依赖、CUDA 不可用和匹配器初始化失败会尽量写成 `status=error`
报告并返回 `1`；在此之前发生的配置、profile 或权重摘要校验失败返回 `2`。

## 报告怎么用

先检查源视频/参考图摘要、配置、profile 和权重身份，再检查解码完整性、两路错误、
实际比较帧数和抽样限制。只有完整的可比较帧才适合统计分歧；缺帧、不可用及失败必须单独
计数。查看分歧时优先检查几何内点数量及分布、背景覆盖、重投影误差、位移、旋转、缩放，
不能用匹配点总数代替机位一致性结论。
尤其 `raw_matches` 并非同一口径：classic 是 KNN 查询组数，LightGlue 是上游返回的
匹配对数（已经过上游置信过滤，尚未经过适配器二次过滤），不能直接比较大小。

分歧不等于改进。视频尚未标注相机 pose 真值时，报告不提供 FP、recall 或准确率提升声明。
CVAT 的异物/异常事件标签也不能自动当作机位移动真值；需要单独人工确认机位状态和变化时段。
这份报告不写入 production readiness，也不代替原有回归或可信 soak 证据。

## 将分歧帧送入 FiftyOne 人工复核

下述原生视频导入是另一条独立离线读取链路：只读取已生成的旁路 JSON 和已有 ReviewBundle，
不重新执行 SceneGuard/LightGlue，不加载权重，不访问原 profile 目录，也不再次解码源视频。
源视频仍须可读以核对媒体摘要；已有 bundle 保存的精确逐帧 PTS 用于核对报告。
准备 bundle 和生成对比报告本身仍按各自流程完整解码，不能省略。

先在恢复后的测试机为同一原视频准备复核包。manifest 的 clip 必须声明正确 camera ID；
视频源和相机共同确定报告归属，不能用文件名猜测。同一个 bundle 内若多个 clip 指向
同一相机和同一视频摘要，报告将因归属歧义而拒绝。每个 clip 一次只能附一份报告；
比较不同配置或权重时应使用新的数据集名字保留独立快照。

以下命令以 manifest 引用上文 `data/regression/original-view.mp4` 为前提，均尚未执行：

```powershell
python scripts/video_review.py prepare --manifest data/regression/scene-review.yaml --output data/review/scene-review.bundle.json
# 使用前节 compare_scene_guard.py 命令生成 original-view.json 后再导入
python scripts/video_review.py fiftyone-import --bundle data/review/scene-review.bundle.json --dataset-name argus-scene-review-01 --scene-comparison data/reports/scene-guard/original-view.json --launch
```

`--scene-comparison` 可以重复，为不同 clip 附加报告。导入仅需要原有 FiftyOne 复核环境，
不需要安装 LightGlue extra、Torch/CUDA 或保留权重路径。省略本选项时保持原有导入行为。
`--launch` 有报告时默认打开 `argus-scene-disagreements`，没有报告时仍打开 `argus-current`；
没有分歧时分歧视图为空是正常结果，并不表示未采样帧已验证。
默认分歧视图不包含执行失败帧；开始下结论前还须检查 `argus-scene-errors` 与报告完整性。

保存视图限定当前完整导入批次，保留原视频上下文：

| 保存视图 | 用途 |
| --- | --- |
| `argus-scene-disagreements` | 两路最终状态不一致的已比较帧 |
| `argus-scene-raw-disagreements` | 两路原始状态不一致的已比较帧 |
| `argus-scene-errors` | 一路未能有效评估的帧；不与正常分歧混算 |
| `argus-scene-unreviewed` | 没有当前报告人工复核确认的已比较帧 |
| `argus-scene-reviewed` | 人工状态合法且确认摘要对应当前报告的帧 |
| `argus-scene-stale-review` | 保留了旧报告人工确认、但摘要不对应当前报告的帧 |
| `argus-current` | 当前完整批次所有源视频帧 |

Argus 报告的 `frame_index=0` 对应 FiftyOne 视频 `frames[1]`，不会把报告抽样序号当作
原视频帧号。界面字段保留原始 PTS/time base 及绝对、相对时间，不以平均 FPS 推算。
定位与复制精确值时使用 `argus_pts_exact` 和 `argus_time_base_exact` 字符串；旧 `argus_pts`
整数仍保留兼容，但超出 JavaScript 安全整数范围时，浏览器显示可能舍入。秒数字段是便于
阅读的浮点值，不应替代精确 PTS/time-base。导入帧 schema 修订后会刷新旧样本字段，保留人工笔记。
这只说明数据字段的绑定，不代表原生播放器已按这些自定义 PTS 定位画面。
已核对的 FiftyOne 官方播放器通过单一 `frameRate` 在时间和帧号之间换算，而不读取
`argus_pts`；参见[播放器定位函数](https://github.com/voxel51/fiftyone/blob/develop/app/packages/looker/src/elements/util.ts)
和[视频组件](https://github.com/voxel51/fiftyone/blob/develop/app/packages/looker/src/elements/video.ts)。
因此，附带旁路报告的 FiftyOne 视频导入在加载 SDK 前，对所有原始帧执行 nominal-FPS
网格预检：相对首帧的精确 PTS 必须接近 `frame_index / fps`。允许误差为半个时间基 tick
（至少 1ns），并保守封顶为 `0.005 / fps` 秒。明显变帧率、累积漂移或粗时间基量化不满足
此条件时明确拒绝，不会静默按平均 FPS 展示成“精确对齐”。纯报告桥接仍能核对 VFR 的
原始 PTS；不附旁路报告的旧导入路径不受此新增限制。
该检查仍不是浏览器视觉验收：容器起点、编辑列表和实际 seek 必须在测试机抽帧核对。
不支持的源使用下文独立受控逐帧图片复核方案；若另外转换视频，必须作为新源重新制作 bundle 和比较报告，
不能修改 fps、复用旧报告或沿用旧人工确认来绕过限制。
最终状态分歧和 raw state 分歧分开显示，原因、几何指标、耗时、错误及报告身份也分别保存。
没有采样的帧不补成一致、stable 或正常；`partial/error` 状态和抽样/截断限制保留可见。
`partial/error` 报告只有能验证 baseline 完整 EOF、源身份、采样集合和精确时间映射时才可
用于复核；零比较帧、解码证据缺失、媒体变化、重复帧或 PTS 不符会在写数据库前拒绝。

bundle 明确声明 `scene_profile_id` 或 `model_version_id` 时，报告中的 profile ID 或声明
模型版本必须对应一致，否则拒绝。未声明时可附加报告中已批准的 profile，其身份单独保留为
诊断信息，不推断运行态授权。这里不要求旧 `scene_reference_path` 的图像摘要等于报告参考图；
它们分别保留来源，不改写原 pipeline evaluation、profile 或 ground-truth 字段。
完整报告 SHA256 位于 `argus_scene_report_sha256`。人工记录使用
`argus_scene_human_state`（`same_pose` / `moved` / `uncertain` / `unreviewable`）、
`argus_scene_human_note` 和 `argus_scene_reviewed_report_sha256`。只有状态合法且复核
摘要与当前帧报告摘要相等时，才视为当前报告已复核；`uncertain/unreviewable` 也只是
“已查看但无法下结论”，不是正确/错误或正负例标签。
旧 `argus_scene_human_state` 文本本身不表示当前报告已复核；应查看上述 reviewed /
unreviewed / stale-review 动态视图。报告 A 换成 B 后，A 的确认保留但失效；恢复逐字相同的
A 时原确认仍适用。同一报告仅移动路径不失效其确认。导入器不写这三个手工字段，避免
其缓存中的空值覆盖另一客户端刚保存的人工记录；这不提供导入和手工编辑之间的事务承诺。
人工复核期间应避免同时重导同一数据集；仅使用 `argus_review.state=complete` 的当前 generation。
报告 SHA256 用于身份追踪和检测导入期间的改动，不是可信签名；导入不重解码核对每帧
tensor SHA256，只保存并显示报告声称的值，不能凭此证明算法确实运行过或诊断声明真实。
`source_verified_against_bundle` 只表示相机、媒体摘要和 PTS 映射与 bundle 相符；
FiftyOne importer 还会重算实际源视频文件摘要，但不会因此独立验证逐帧图像摘要。
只使用来自受控对比流程的报告。

FiftyOne App 主要用于定位、播放和可视化复核；本轮尚未验证目标版本 App 对任意标量帧字段
的直接编辑支持。测试机验收时可通过其 Python API 保存人工记录，例如先在 App 记下
当前批次 sample ID 和原始帧号，再在同一复核环境中执行：

```python
import fiftyone as fo

dataset = fo.load_dataset("argus-scene-review-01")
sample = dataset["<current-generation-sample-id>"]
original_frame_index = 0  # 替换为 App 中待复核的原始帧号，不是抽样序号
frame = sample.frames[original_frame_index + 1]
frame["argus_scene_human_state"] = "uncertain"
frame["argus_scene_human_note"] = "遮挡严重，无法确认相机是否移动"
frame["argus_scene_reviewed_report_sha256"] = frame["argus_scene_report_sha256"]
frame.save()
```

示例 sample ID 须替换，原始帧号须人工核对；只对已附报告的当前批次帧复核。重导相同报告保留
人工记录；报告变化时保留旧文本，但旧摘要不能被当作新报告已经复核。人工复核字段不
回写 CVAT/Argus 标签，不自动成为 pose 真值，也不会批准新场景、模型或恢复生产放行。
更完整的复核环境说明见 [CVAT / FiftyOne 集成手册](video-review-integrations.md)。

## VFR 受控逐帧图片复核

原生视频模式无法满足 PTS/FPS 网格时，可显式改用独立图片模式。原生模式仍拒绝该输入；
新模式不改视频、不把它转成 CFR、不改 bundle 时间戳，也不进入生产 SceneGuard。
下面命令仍是测试/复核机器恢复后的待验收步骤，不是在本机执行的命令。

```powershell
python scripts/video_review.py scene-images-prepare --bundle data/review/scene-review.bundle.json --scene-comparison data/reports/scene-guard/original-view.json --output-dir data/review/scene-images-01
python scripts/video_review.py scene-images-import --bundle data/review/scene-review.bundle.json --scene-comparison data/reports/scene-guard/original-view.json --manifest data/review/scene-images-01/manifest.json --dataset-name argus-scene-images-01 --launch
```

两个命令均要求显式报告，可重复 `--scene-comparison` 为不同 clip 绑定报告。准备命令只读
源视频和报告，使用严格 FFmpeg/ffprobe 完整顺序解码，不按 FPS seek，不重新执行任一算法。
它用原始帧号定位，核对精确 PTS/time base 及报告中的 BGR tensor SHA256，输出原尺寸
无损 PNG 和来源 manifest。尾部也必须解码到干净 EOF；错误不能被选中帧已提取掩盖。
manifest 保留原始帧号、精确 PTS、源视频/PNG/tensor/报告摘要及报告状态、摘要和抽样限制。
这条链验证的是图像与已绑定诊断报告的对应关系，不证明该报告来自可信算法执行。

默认选择最终/raw 分歧和错误帧；`--selection all-compared` 可改为全部已比较帧。
未比较帧不补为 stable 或正常。默认且硬性上限 10000 张图片、10 GiB，可通过正整数
`--max-images` / `--max-output-bytes` 显式降低；超限失败，不截断或缩图。
输出必须是新的独立目录，没有覆盖开关；输入路径和受保护 profile 目录不能作为输出。
准备步骤需 NumPy 和 PATH 中的 FFmpeg/ffprobe，但不需要 LightGlue、Torch/CUDA
或模型权重。详细环境边界见 [集成手册](video-review-integrations.md#vfr-源的受控逐帧图片复核)。

图片导入在连接 FiftyOne 前重新校验 manifest、PNG、报告和实际源视频摘要，仍需要这些
文件可读；不会重新解码视频或运行模型。一张 PNG 对应一个 image sample，不使用
`sample.frames` 或视频播放器的 FPS 换算。`argus_frame_index` 是原视频从 0 开始的帧号，
`argus_frame_number` 是原始帧号加 1（只是元数据，不是图片列表序号），精确值使用
`argus_pts_exact` / `argus_time_base_exact`。报告摘要在 `argus_scene_report_sha256`，
诊断摘要与逐帧结果分别保留在 sample 的 JSON 字段中。

图片模式使用独立数据集名字，不得沿用上文原生视频数据集；`--launch` 打开
`argus-scene-unreviewed`。current、分歧、raw 分歧、错误、reviewed 和 stale-review
视图都在 sample 级筛选。报告更新后新建图片目录，再重导同一个图片数据集；相同源帧的
人工笔记和旧确认摘要保留，但摘要不匹配新报告时仍须人工重看。
FiftyOne 没有全局导入事务，导入或失败恢复期间保存视图可能短暂显示子集；必须同时核对
`dataset.info["argus_scene_images"]["state"] == "complete"`。`importing/failed` 或空/部分
视图都不能当整批成功；数据库不可写时，pending 隔离补偿也可能失败。这些故障恢复行为
尚待真实复核环境验收，不表示本轮已经验证通过。每次导入和失败重试都必须新建独立 CLI
进程；已加载 SDK 的人工笔记会话或同进程第二次导入会被拒绝。导入前先在人工会话保存笔记。
图片导入首次加载 SDK 前禁用单例缓存和数据库管理权限，并直接预检实际 SDK JSON 配置，
拒绝含非空或非法 `module_path` 钩子的配置；仅清空环境变量不能覆盖配置文件里的钩子。
要求 FiftyOne `>=1.8,<2`，且实际配置必须支持禁用；导入期间不要并发修改 SDK 配置。
数据库、数据集及锁 collection 必须使用 `PRIMARY`、显式 `readConcernLevel=local` 和有确认的写入；
未显式指定级别（`None`）、`majority/snapshot`、从节点读取或 `w=0` 连接失败关闭。主节点的
majority 视图也可能暂未包含前次 `w=1` 已确认的登记，不能只改一次 raw 查询而保留 SDK 旧策略。
在专用数据库原 URI 中明确设置 `readConcernLevel=local`，保留认证、TLS 等其他配置后开新 CLI。
不会硬刷新既有对象，也不替用户保存或丢弃草稿。
以下仅是另一个复核会话的人审操作示例，不要在此会话调用图片导入：

```python
import fiftyone as fo

dataset = fo.load_dataset("argus-scene-images-01")
sample = dataset["<current-generation-image-sample-id>"]
assert sample["argus_frame_index"] == 0  # 人工核对待复核的原视频帧号
sample["argus_scene_human_state"] = "uncertain"
sample["argus_scene_human_note"] = "光照变化，无法确认机位是否移动"
sample["argus_scene_reviewed_report_sha256"] = sample["argus_scene_report_sha256"]
sample.save()
```

sample ID、帧号和记录均须人工替换确认，不能批量自动盖上当前报告 SHA。
合法人工状态仍为 `same_pose/moved/uncertain/unreviewable`，后两者只表示已查看、不能下结论。
动态 reviewed/stale-review 视图以当前报告 SHA 和合法人工状态判断，不因旧 human_state
文本存在就当新报告已复核。导入器不写人工字段；避免导入与手工修改并发进行。
新报告未再选中的旧图片保留为历史样本，不在当前批次视图中；要查看当前报告的全部比较帧，
重新准备时显式选择 `--selection all-compared`，不要从分歧集合消失推断已复核或已通过。
首次补齐 clip 的 split/采集批次后，未选中历史图片的未分配值按同一原始媒体身份兼容检查，
不因此阻断重试；已分配身份不可改绑，跨 clip 的视频/采集批次 split 隔离仍生效。
带版本的源身份登记会保留零选中、缺席 clip 的约束，在写样本前持久化，失败不清除已保存
绑定；新数据集首次保存所有权时就保存完整 v2 登记。旧数据集缺登记或仍为 v1 时不自动迁移，
因为无法证明历史零图、无报告或缺席来源的完整身份，v1 也不能证明是否由不完整自动迁移产生。
保留旧数据和备注，使用新集合，或另行制定有可信完整历史凭据的显式迁移方案；不能仅修改
schema 字符串绕过检查。登记损坏、超限或历史图片所属 clip 缺席于 v2 登记也拒绝导入，
不从历史图片自动重建登记或解除绑定；尚无历史图片的当前新 clip 可正常登记。
首次登记提交前失败若留下无可信 owner 的集合，不会自动接管或删除，应换名字创建；提交后
确认丢失则在新 CLI 进程重新读取持久登记重试，仍保留所有已提交约束，不让失败进程的脏缓存
借 SDK 加载补提交。只读元数据预检先于数据集加载，旧 v1、无 owner 或不符 owner/suite/host/
媒体类型的目标直接拒绝；加载后再次核对持久身份与新对象，不使用缓存硬刷新。
数据库集合 `argus_scene_image_import_locks` 提供按数据集名唯一的跨 CLI 互斥，获取后才能
读取或写入目标，结束时仅删除自己 token 的记录，覆盖不同用户/临时目录绕过文件锁的情况。
启用前先停止旧版本 CLI，不能混跑未参与数据库锁的旧导入器，也不能并发直接改写受管登记。
插锁前须确认普通集合和默认 `_id_` 唯一索引，拒绝时序/capped/view/clustered/TTL 或无法检查的
存储；集合及所有索引 collation 仅允许缺省或 `{locale: "simple"}`，确保 token 二进制精确比较。
不自动重建或清空。仅支持独立 mongod/普通副本集，拒绝 mongos，禁止直连 shard 节点。
锁没有 TTL，不自动偷锁；遇持久层/传输异常、超时或中断保留锁并停止补偿写，防止迟到提交
与新导入交错；补偿阶段被捕获的未知写失败、错误分类器自身失败同样保锁。
主失败与释放失败会同时保留诊断类型；无法转换 SDK 错误文本时 CLI 使用固定占位。
遗留锁恢复须暂停新入口并确认客户端及相关服务端操作均已结束；PID 消失或 `killOp` 成功不够。
管理员须以 PRIMARY/显式 local 重读，只能按 `_id + 当前原 token` 并显式 simple 比较条件删除；
删除确认丢失时先重读并核查服务端，不盲重试，
不清理后来者的锁。详见[人工恢复流程](video-review-integrations.md#图片导入遗留锁的人工恢复)。
互斥不代替样本/视图事务或数据库持久性；副本集应配置多数确认写入并验收切换。
数据集版本与 SDK 不一致时拒绝自动迁移，需管理员另行处理；使用专用复核数据库，SDK 初始化
可能写入自身数据库配置，退出时也可能清理非持久集合，这不是任意 SDK 副作用的沙箱隔离。
这不是原视频时序回放、pose 真值、profile 批准或生产发布证据；不改变现有放行和恢复逻辑。

测试机恢复后至少覆盖：原位、光照变化、局部遮挡、平移、旋转、变焦、强视差、纹理不足、
错误场景、断流录制片段、恢复原位、损坏视频、缺依赖、错误权重摘要及 CUDA 不可用。
核对任一种输入都没有写入生产 profile 或改变原有放行结果；先跑对应单元测试，再做真实
CUDA 权重和完整语料回放。CLI 及 matcher/geometry/runner 回归源码已经提供，尚未执行。
额外检查：负起始/大整数 PTS、稀疏采样、受支持 CFR 的浏览器逐帧 seek、不支持 VFR
在 SDK 前拒绝；报告 A→B→移除→原样 A、缺失/空复核字段、并发保存笔记和导入 schema 升级。
图片模式额外核对真实 VFR 原始帧/PTS/图像摘要、完整 EOF、预算超限、受保护目录和
输出已存在、文件篡改、A→B→原样 A 的 sample 级人工确认及独立数据集所有权；
原生视频模式仍应拒绝不支持的 VFR。另验收旧 v1/部分缺失 v2 拒绝、首次登记提交前后失败、
补偿失败后的新进程重试、预加载 SDK 及隐式缓存写入拒绝、人审笔记不变。另覆盖配置文件钩子
预拒绝、非 PRIMARY/无确认写拒绝、跨临时目录并发锁及锁获取/释放失败。所有新测试源码尚未执行。
再覆盖不安全锁集合预拒绝、服务端迟到提交、未知补偿写保锁、按原 token 恢复及双失败诊断脱敏。
另覆盖非 simple/非法 collation、SDK/锁未显式 local 的拒绝、分类器自身失败保锁和异常文本诊断降级。

## 对抗审查补充

- 除关键点之外，提取描述子和图像尺寸也须校验结构与有限值；坏描述子不能被空匹配掩盖。
- CUDA 诊断核对返回张量的实际设备，不只记录请求的设备名；不一致是显式错误。
- 生产模块的导入隔离守门覆盖普通导入、别名和常见动态导入写法，但静态检查不是任意
  Python 动态代码的形式化证明。当前入口仍限定为独立离线命令。
- 本轮保留原 SceneGuard、pipeline、默认配置和发布证据验证代码不变；运行时影响及资源
  竞争仍须在测试机验证，不能由源码审查代替。
