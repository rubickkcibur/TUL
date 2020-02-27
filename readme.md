所有代码与原数据用户分类文件夹同级，000，001....181

数据处理：

先用data process.py对原数据截断，仅保留经纬度，小数点后保留3位

用encode.py将经纬度用数字表示

用encode_path.py将原数据的路径的节点从经纬度表示到数字表示

最终数字节点表示的路径文件放在out/文件夹下(已附带)

用embbeding.py生成每个节点的embbeding

embbeding文件是embbeding.model

运行代码：

util.py里面是加载数据，每个用户取前30%做train集，后70%做test

BiLSTM.py里是RNN模型，照抄的样例文章的code

main文件train，每个iter之后测试

只测了top1,top5,top10准确率