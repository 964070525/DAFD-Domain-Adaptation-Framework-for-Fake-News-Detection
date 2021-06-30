
from wordcloud import WordCloud
import matplotlib.pyplot as plt

text = open("./data/po.txt", encoding='utf8').read()
text = text.replace('\n', "").replace("\u3000", "")

# 分词，返回结果为词的列表
text_cut = text.split(" ")
# 将分好的词用某个符号分割开连成字符串
text_cut = ' '.join(text_cut)
stop_words = open("./data/stop.txt", encoding="utf8").read().split("\n")

# 使用WordCloud生成词云
word_cloud = WordCloud(font_path="simsun.ttc",  # 设置词云字体
                       background_color="white",  # 词云图的背景颜色
                       stopwords=stop_words)  # 去掉的停词
word_cloud.generate(text_cut)

# 运用matplotlib展现结果
plt.subplots(figsize=(12, 8))
plt.imshow(word_cloud)
plt.axis("off")
plt.show()
