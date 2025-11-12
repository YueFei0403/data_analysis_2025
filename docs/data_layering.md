1. ODS (Operational Data Store)-- 操作数据层
    (1) 原始业务系统导入的明细数据（最接近源系统）
    (2) 存放未经处理的原始数据，作为“事实依据”
2. DWD (Data Warehouse Detail) -- 明细数据层
    (1) 经过清洗，脱敏，规范化处理的细粒度明细数据
    (2) 保留业务事实，保证一致性与可追溯性
3. DWM (Data Warehouse Middle) -- 中间汇总层
    (1) 对DWD层数据进行汇总，聚合，维度扩展
    (2) 形成可复用的分析中间表
4. DWS (Data Warehouse Service) -- 服务数据层
    (1) 面向主题域（如用户、订单、营销）的宽表
    (2) 提供直接给分析模型或报表使用的统一视图
5. ADS （Application Data Store）-- 应用数据层
    (1) 针对具体应用（如仪表盘、推荐系统）定制的数据
    (2) 直接支撑BI分析，算法训练或产品接口