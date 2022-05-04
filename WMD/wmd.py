import jieba
from gensim.models import Word2Vec
from gensim.similarities import WmdSimilarity
import gensim
from LDA_Similarity.ApiDetailedDescription import read_APIDetailedDescription
from LDA_Similarity.data_preprocess import preprocess_data_en
from nltk.corpus import stopwords


# 加载停用词
def get_stop_words(stop_words_dir):
    stop_words = []

    with open(stop_words_dir, 'r', encoding='utf-8') as f_reader:
        for line in f_reader:
            line = delete_r_n(line)
            stop_words.append(line)

    stop_words = set(stop_words)
    return stop_words


# 去掉空字符
def delete_r_n(line):
    return line.replace('\r', '').replace('\n', '').strip()


# 数据预处理
def wmd_sim(Source_Mashup):
    id_keys = list(Source_Mashup.keys())
    pdata = []
    corpus = []
    documents = []
    for i in id_keys:
        pdata += preprocess_data_en([Source_Mashup[i][1]])

    for each in pdata:
        text = list(each.replace('\n', '').split(' '))
        #print(text)
        corpus.append(text)
    print(len(corpus))
    print(20 * '*', '加载模型', 40 * '*')
    pretrain_model_path = r'../data/pre_trained_embeddings/GoogleNews-vectors-negative300.bin'
    embedding = gensim.models.KeyedVectors.load_word2vec_format(pretrain_model_path, binary=True)
    #model = Word2Vec.load(pretrain_model_path)
    print("加载完成")
    num_best = 20
    instance = WmdSimilarity(corpus, embedding, num_best=num_best)
    return instance

if __name__ == '__main__':
    BIAODIAN = ['', ',', '(', ')', '[', ']', ' ','.']
    sent = 'B+树是一种树数据结构，叶子结点存储关键字以及相应记录的地址，叶子结点以上各层作为索引使用。'
    Souce_mashup = {}
    read_APIDetailedDescription(Souce_mashup)
    sent = Souce_mashup[0][1]
    sent_w = list(jieba.cut(sent))
    sent_w = [w for w in sent_w if w not in BIAODIAN]
    english_stopwords = stopwords.words('english')
    query = [w for w in sent_w if not w in english_stopwords]

    # 在相似性类中的“查找”query
    source_mashup = {}
    read_APIDetailedDescription(source_mashup)
    instance = wmd_sim(source_mashup)
    sims = instance[query]
    # 返回相似结果
    print('source_Query:')
    print(sent)
    for i in range(20):
        print('sim = %.4f' % sims[i][1])
        print('sims[i][0]:', sims[i][0])
        print(source_mashup[sims[i][0]])