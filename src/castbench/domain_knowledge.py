"""Frozen human-reviewed domain knowledge, separate from empirical test outcomes."""

from __future__ import annotations

VERSION = "2026-09-20.1"

# Search metadata only: no new physical claim or forecast coefficient is inferred.
SEARCH_TERMS = {
    "天气": "weather meteorology",
    "气温": "temperature",
    "温度": "temperature",
    "雨": "rain precipitation",
    "降水": "precipitation rainfall",
    "雪": "snow snowfall",
    "寒冷": "cold",
    "低温": "cold temperature",
    "高温": "heat temperature",
    "风速": "wind speed",
    "大风": "strong wind gusts",
    "风资源": "wind resource power",
    "风向": "wind direction",
    "风切变": "wind shear",
    "结冰": "icing ice",
    "功率": "power generation",
    "负荷": "load electricity demand",
    "通勤": "commuting commute",
    "骑行": "bike cycling ridership",
    "日历": "calendar",
    "工作日": "working day weekday",
    "休息日": "rest day weekend",
    "假日": "holiday",
    "节日": "holiday",
    "夏令时": "daylight saving time DST",
    "辐射": "solar radiation irradiance",
    "组件": "photovoltaic panel",
    "沙尘": "dust",
    "湿度": "humidity",
    "云": "cloud",
    "单位": "unit conversion",
    "聚合": "aggregation",
    "时钟": "clock time",
    "未来": "future visibility",
    "模型": "model forecast",
    "样本库": "training case library",
    "证据": "evidence",
    "重复": "duplicate",
    "取暖": "heating",
    "制冷": "cooling",
    "度时": "degree hours",
    "站点": "station",
    "网格": "grid",
    "容量": "capacity",
}


def with_search_metadata(entry):
    result = dict(entry)
    text = entry["topic"] + " " + entry["text"]
    aliases = set(entry.get("search_aliases", []))
    aliases.update(value for term, value in SEARCH_TERMS.items() if term in text)
    aliases.update(name.replace("_", " ") for name in entry.get("variables", []))
    result.update(language="zh-CN", search_aliases=sorted(aliases))
    return result


SOURCES = {
    "uci": "https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset",
    "noaa": "https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database",
    "weather": "https://open-meteo.com/en/docs/historical-weather-api",
    "eia": "https://www.eia.gov/outlooks/aeo/IIF_weather/",
    "pv": "https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.pvsystem.pvwatts_dc.html",
    "snow": "https://samrepo.nlr.gov/help/pvwatts_system_design.html",
    "wind": "https://www.energy.gov/cmei/wind/articles/how-do-wind-turbines-survive-severe-weather-and-storms",
}

# level, topic, independently stated fact/hypothesis, source, epistemic status.
COMMON = [
    (
        "boundary",
        "已知未来情境",
        "当前场景允许读取给定的未来外生变量，但不能读取本样本的未来目标。它是已知未来情境的能力评测，不等于真实天气预报误差下的生产预测。",
        "用户确认的评测协议",
        "benchmark_assumption",
    ),
    (
        "boundary",
        "训练样本库",
        "样本库截止于首个测试目标日前。后续测试样本的真实历史可作为该样本输入，但不得追加到冻结样本库。",
        "CastBench冻结评测协议",
        "benchmark_assumption",
    ),
    (
        "expert",
        "证据不是指令",
        "知识说明可能的影响机制，不提供固定修正幅度。同类事件的影响可能被季节、当前水平和其他因素抵消，应综合工具证据决定。",
        "CastBench专家知识使用规范",
        "methodological_guidance",
    ),
    (
        "boundary",
        "重复信息",
        "派生事件、原始天气及其多个等价表示不是独立证据；不能因同一因素出现在多列就重复放大修正。",
        "字段构造说明与统计原则",
        "methodological_guidance",
    ),
    (
        "boundary",
        "时钟与聚合",
        "每日广播值不是每小时新增量；滚动累计也不能逐行相加。优先遵守字段单位、时间覆盖和来源说明。",
        "weather",
        "definition",
    ),
    (
        "boundary",
        "模型工具",
        "基础模型只是可选工具，没有必须采用的初始预测。模型只读取历史目标序列；当前所有外生变量由Agent侧工具使用。",
        "用户确认的模型输入协议",
        "benchmark_assumption",
    ),
]

DOMAIN = {
    "bike": [
        (
            "foundation",
            "任务",
            "目标是每小时骑行次数，不是道路流量、站点库存或未满足需求。供给不足可能限制观测次数，但本表没有库存证据，不能自动断言缺车。",
            "uci",
            "definition",
        ),
        (
            "foundation",
            "温度编码",
            "小时级气温用47*temp-8恢复摄氏度，体感温度用66*atemp-16恢复；hum*100是相对湿度百分比，windspeed*67是km/h。",
            "uci",
            "definition",
        ),
        (
            "foundation",
            "日历",
            "weekday为0周日到6周六，workingday排除周末与假日，season为1冬2春3夏4秋；不能用月份猜测工作日。",
            "uci",
            "definition",
        ),
        (
            "expert",
            "通勤与休闲",
            "工作日可能呈通勤峰，非工作日可能偏日间休闲。先看历史曲线及日历变化，再判断峰时是否适合平移；一天历史不足以识别稳定周规律。",
            "领域机制假设；参考UCI工作日字段设计",
            "hypothesis",
        ),
        (
            "expert",
            "天气舒适性",
            "雨雪、寒冷、高温或大风可能抑制骑行，但温和升温也可能增加需求。不可假定全年统一温度斜率；变化幅度需训练样本支持。",
            "领域机制假设；不使用测试目标估计系数",
            "hypothesis",
        ),
        (
            "measurement",
            "降水量和现象",
            "precip_1h_mm是日总量约束的小时分配；rain_flag是天气现象。分配标志为1不代表该小时有独立雨量观测。",
            "本地Bike v5报告",
            "definition",
        ),
        (
            "measurement",
            "天空遮蔽",
            "sky_obscured_flag表示云量难以观测，不是9/8云量。cloud_valid_hours低时，日平均云量的代表性较弱。",
            "本地Bike v5报告；noaa",
            "definition",
        ),
        (
            "measurement",
            "多来源天气",
            "UCI天气等级和NOAA雨雪雷暴来自不同来源。表面不一致可反映观测地点、时间口径或编码差异，不能仅靠不一致判定哪列错误。",
            "本地Bike v5报告",
            "definition",
        ),
        (
            "boundary",
            "夏令时",
            "采用America/New_York本地时钟，春季缺02时、秋季只保留第二个01时。24点整日测试不跨这些不完整或实际间隔不规则日期。",
            "本地Bike v5报告",
            "definition",
        ),
        (
            "boundary",
            "范围",
            "骑行次数非负；预测均值可以是小数，不强制取整，也不设置从测试最大值推导的上限。",
            "目标定义及评测协议",
            "methodological_guidance",
        ),
    ],
    "gef": [
        (
            "foundation",
            "任务和时标",
            "预测匿名Zone 6小时负荷。保留源hour-ending标签，00时负荷表示前一小时。窗口按显示日期00至23时组织；新版所有日级事件按显示日期定义，不能把它们再平移一小时，也不能把源标签称为区间开始。",
            "本地GEF报告",
            "definition",
        ),
        (
            "foundation",
            "匿名空间",
            "11个温度站和Zone 6没有已确认坐标。站点编号不是城市名称，Virginia强天气是州级代理，不证明负荷区正好遭遇该事件。",
            "本地GEF报告",
            "definition",
        ),
        (
            "expert",
            "取暖与制冷",
            "冷暖变化可分别影响取暖和制冷需求，单一全年线性温度相关容易掩盖机制。方向还取决于季节、时段和用电结构。",
            "eia",
            "mechanism",
        ),
        (
            "expert",
            "度时",
            "65F可作为透明的冷热参考点计算度时，但它不是Zone 6已识别的平衡温度，也不能直接把度时换算成负荷修正量。",
            "eia",
            "definition",
        ),
        (
            "expert",
            "日历形态",
            "休息日可能减少商业通勤相关用电，同时住宅活动可能改变时段。应比较日内形态，不能把所有节日都视为固定比例下降。",
            "用电结构机制假设",
            "hypothesis",
        ),
        (
            "measurement",
            "温度站群",
            "多站均值反映区域热状态，站间离散保留空间差异。多列同向不等于多个独立升降温证据。",
            "本地GEF字段定义",
            "definition",
        ),
        (
            "measurement",
            "湿度变量",
            "温度、露点、相对湿度与VPD相互关联；结合使用可描述湿热背景，但不能机械叠加四次天气效应。",
            "weather",
            "definition",
        ),
        (
            "measurement",
            "波次事件",
            "热浪和寒潮严重度由季节温度规则派生，不是独立观测。本版固定沿用提供的事件，不能声称其阈值仅由训练段估计。",
            "本地GEF报告",
            "definition",
        ),
        (
            "boundary",
            "未来可见约定",
            "NOAA和再分析天气在本评测作为用户指定的已知情境开放；真实日前预测中通常不具备相同可用性。",
            "用户确认；本地GEF报告",
            "benchmark_assumption",
        ),
        (
            "boundary",
            "负荷单位",
            "源报告未说明Zone 6的绝对功率单位。保留原始数值标度，禁止擅自称为MW或将其换算为电量。",
            "本地GEF报告",
            "definition",
        ),
    ],
    "sunny": [
        (
            "foundation",
            "净功率",
            "目标为给定标度的场站功率，负数保留。辅助用电是可能解释之一，未有设备记录不能认定全部负值都由其造成。",
            "用户确认的源数据语义",
            "benchmark_assumption",
        ),
        (
            "foundation",
            "辐射分量",
            "GHI是水平总辐射，DHI是水平散射，direct_radiation是水平直射，DNI是法向直射。不能把DNI直接与DHI相加解释为GHI。",
            "weather",
            "definition",
        ),
        (
            "expert",
            "辐射与形态",
            "入射辐射是核心资源。云层变化可能影响日内峰形和爬坡；外部水平辐射不是组件平面辐射，不能直接换算功率。",
            "pv",
            "mechanism",
        ),
        (
            "expert",
            "温度",
            "组件功率与组件温度有关，组件温度还受辐射、风和安装条件影响。只有环境气温时不应假定精确组件降额系数。",
            "pv",
            "mechanism",
        ),
        (
            "expert",
            "积雪",
            "地面积雪代理提示覆盖风险，但组件积雪还与倾角、温度和滑落有关；不能按有雪就整日停发。",
            "snow",
            "mechanism",
        ),
        (
            "expert",
            "沙尘",
            "空气沙尘可能衰减辐射，表面积灰可能造成额外损失；当前字段没有直接测量组件积灰，已反映在辐射中的衰减不要重复扣减。",
            "领域机制假设与字段语义",
            "hypothesis",
        ),
        (
            "measurement",
            "站点与网格",
            "三站辐射中心和离散度与网格辐射反映不同空间尺度。差异不是天气错误的充分证据，先检查共同的强弱与局地变动。",
            "本地Sunny报告",
            "definition",
        ),
        (
            "measurement",
            "时间分辨率",
            "部分天气是小时值在15分钟上保持。按0.25小时积分可得到相应能量代理；不能把小时日照秒数重复累加四次。",
            "本地Sunny报告；weather",
            "definition",
        ),
        (
            "boundary",
            "雪与结冰",
            "snowfall_24h_sum是重叠的24小时累计，不是每15分钟新增降雪。icing_risk_score是条件比例，不是结冰概率或确认损失。",
            "本地Sunny报告",
            "definition",
        ),
        (
            "boundary",
            "缺少设备参数",
            "容量、倾角、方位和绝对功率单位未确认，不套用精确PVWatts容量模型。缺少这些信息不妨碍相对情境分析。",
            "pv",
            "methodological_guidance",
        ),
    ],
    "windy": [
        (
            "foundation",
            "净功率",
            "保留提供的负功率，不做非负裁剪。辅助用电或运行状态可能影响读数，但没有场站日志不能反推具体故障。",
            "用户确认的源数据语义",
            "benchmark_assumption",
        ),
        (
            "foundation",
            "风资源",
            "风速与风功率非线性相关，但电功率还受额定平台、控制和可用率影响。v的三次方只是资源代理，不是全年通用功率曲线。",
            "wind",
            "mechanism",
        ),
        (
            "expert",
            "大风",
            "超过设备切出条件可能保护停机，但阈值随设备变化。没有机型与运行信息时，不根据通用25m/s等阈值断言本场停机。",
            "wind",
            "mechanism",
        ),
        (
            "expert",
            "风切变",
            "不同高度风速有助于区分垂直结构；80与120m估计的幂律指数是当时的描述，不宜被当作全年固定系数。",
            "字段物理定义和幂律关系",
            "definition",
        ),
        (
            "expert",
            "风向",
            "风向是圆周变量，359与1度应接近0度。时间集中度与多站方向一致性不是同一个指标，不能混用。",
            "圆统计恒等式；本地Windy报告",
            "definition",
        ),
        (
            "expert",
            "结冰",
            "低温、高湿和降水构成结冰背景，但风险分数不是实测叶片结冰。可能降低效率，不能仅凭其中一个条件就按停机修正。",
            "本地Windy风险字段定义",
            "hypothesis",
        ),
        (
            "measurement",
            "单位边界",
            "外部风速km/h除3.6转换m/s；原站点风速报告只写原单位，未确认前不能与m/s数值直接比较或套用物理切出阈值。",
            "本地Windy报告",
            "definition",
        ),
        (
            "measurement",
            "两种融合风速",
            "矢量融合可能因方向相反抵消；立方等效风速体现强风权重。二者回答不同问题，不能把差值直接当作功率误差。",
            "本地Windy报告",
            "definition",
        ),
        (
            "measurement",
            "小时保持与天气码",
            "小时天气在15分钟上保持不代表每刻有独立观测；WMO天气码是类别，不能做线性大小解释。",
            "weather",
            "definition",
        ),
        (
            "boundary",
            "相近场站",
            "Windy与Sunny共享部分外生变量是相近区域设计，不是文件错误；评测时不能把它们当作完全独立地理区域的泛化证据。",
            "用户确认的场景关系",
            "benchmark_assumption",
        ),
    ],
}


def knowledge_for(domain, variables):
    entries, catalog = [], []
    for i, (level, topic, text, source, status) in enumerate(COMMON + DOMAIN[domain]):
        sid = f"{domain}.knowledge.{i:03d}"
        source = SOURCES.get(source, source)
        entries.append(
            dict(
                entry_id=sid,
                topic=f"{level}/{topic}",
                text=text,
                source=source,
                version=VERSION,
                scope="general" if i < len(COMMON) else "domain",
                variables=[],
            )
        )
        catalog.append(dict(entry_id=sid, level=level, epistemic_status=status))
    for v in variables:
        sid = f"{domain}.variable.{v['name']}"
        entries.append(
            dict(
                entry_id=sid,
                topic=f"variable/{v['name']}",
                text=f"{v['description']}；单位：{v['unit']}；来源：{v['lineage']}。",
                source=v["lineage"],
                version=VERSION,
                scope="domain",
                variables=[v["name"]],
            )
        )
        catalog.append(dict(entry_id=sid, level="variable", epistemic_status="definition"))
    return [with_search_metadata(entry) for entry in entries], catalog
