import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_cohere.embeddings import CohereEmbeddings  
from langchain_community.vectorstores import FAISS
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import cohere
import requests



# -------------------------------------------------------
from collections import Counter
from keras.models import load_model
import re
import numpy as np
import string
from nltk import ngrams
import nltk
from unidecode import unidecode
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, Activation, Bidirectional
from tensorflow.keras.optimizers import Adam

"""
Đọc dữ liệu từ file corpus.txt, mỗi dòng trong file đại diện cho một câu hoặc đoạn văn bản.
Chi tiết:
corpus_file_path: Đường dẫn đến file corpus.
Sử dụng with open(...) để mở file trong chế độ đọc với mã hóa UTF-8.
Duyệt từng dòng trong file, loại bỏ khoảng trắng đầu và cuối bằng strip(), sau đó lưu vào danh sách data.
In ra số lượng câu dữ liệu đã đọc.
"""
# pip install -U scikit-learn
# corpus_file_path = './corpus.txt'
# # Đọc dữ liệu từ file corpus.txt
# with open(corpus_file_path, 'r', encoding='utf-8') as f:
#     data = [line.strip() for line in f]  # Sử dụng strip() để loại bỏ khoảng trắng hoặc ký tự xuống dòng
# print('Số câu dữ liệu:', len(data))

"""
vowel: Danh sách các nguyên âm tiếng Việt bao gồm cả chữ hoa và chữ thường, cũng như các dấu phụ.
full_letters: Kết hợp vowel với danh sách các phụ âm tiếng Việt.
"""
vowel = list('aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴ')
full_letters = vowel + list('bcdfghjklmnpqrstvwxzBCDFGHJKLMNPQRSTVWXZđĐ')

"""
Mỗi ký tự đặc biệt (có dấu) được ánh xạ tới một chuỗi ký tự tương ứng. Ví dụ, 'á' được ánh xạ thành 'as'
"""
typo = {
    'á': 'as', 'Á': 'As', 'à': 'af', 'À': 'Af', 'ả': 'ar', 'Ả': 'Ar', 'ã': 'ax', 'Ã': 'Ax', 'ạ': 'aj', 'Ạ': 'Aj',
    'ắ': 'aws', 'Ắ': 'Aws', 'ằ': 'awf', 'Ằ': 'Awf', 'ẳ': 'awr', 'Ẳ': 'Awr', 'ẵ': 'awx', 'Ẵ': 'Awx', 'ặ': 'awj', 'Ặ': 'Awj', 'ă': 'aw', 'Ă':'Aw',
    'ấ': 'aas', 'Ấ': 'Aas', 'ầ': 'aaf', 'Ầ': 'Aaf', 'ẩ': 'aar', 'Ẩ': 'Aar', 'ẫ': 'aax', 'Ẫ': 'Aax', 'ậ': 'aaj', 'Ậ': 'Aaj', 'â': 'aa', 'Â': 'Aa',
    'é': 'es', 'É': 'Es', 'è':'ef', 'È': 'Ef', 'ẻ': 'er', 'Ẻ': 'Er', 'ẽ': 'ex', 'Ẽ': 'Ex', 'ẹ': 'ej', 'Ẹ': 'Ej',
    'ế': 'ees', 'Ế': 'Ees', 'ề': 'eef', 'Ề': 'Eef', 'ể': 'eer', 'Ể': 'Eer', 'ễ': 'eex', 'Ễ': 'Eex', 'ệ': 'eej', 'Ệ': 'Eej', 'ê': 'ee', 'Ê': 'Ee',
    'í': 'is', 'Í': 'Is', 'ì': 'if', 'Ì':'If', 'ỉ': 'ir', 'Ỉ': 'Ir', 'ĩ': 'ix', 'Ĩ': 'Ix', 'ị': 'ij', 'Ị': 'Ij',
    'ó': 'os', 'Ó': 'Os', 'ò': 'of', 'Ò': 'Of', 'ỏ': 'or', 'Ỏ': 'Or', 'õ': 'ox', 'Õ': 'ox', 'ọ': 'oj', 'Ọ': 'Oj',
    'ố': 'oos', 'Ố': 'Oos', 'ồ': 'oof', 'Ồ': 'Oof', 'ổ': 'oor', 'Ổ': 'Oor', 'ỗ': 'oox', 'Ỗ': 'Oox', 'ộ': 'ooj', 'Ộ': 'Ooj', 'ô': 'oo', 'Ô': 'Oo',
    'ớ': 'ows', 'Ớ': 'Ows', 'ờ': 'owf', 'Ờ': 'OwF', 'ở': 'owr', 'Ở': 'Owr', 'ỡ': 'owx', 'Ỡ': 'Owx', 'ợ': 'owj', 'Ợ': 'Owj', 'ơ': 'ow', 'Ơ': 'Ow',
    'ú': 'us', 'Ú': 'Us', 'ù': 'uf', 'Ù': 'Uf', 'ủ': 'ur', 'Ủ': 'Ur', 'ũ': 'ux', 'Ũ': 'Ux', 'ụ': 'uj', 'Ụ': 'Uj',
    'ứ': 'aws', 'Ứ': 'Uws', 'ừ': 'uwf', 'Ừ': 'Uwf', 'ử': 'uwr', 'Ử': 'Uwr', 'ữ': 'uwx', 'Ữ': 'Uwx', 'ự': 'uwj', 'Ự': 'Uwj', 'ư': 'uw', 'Ư': 'Uw',
    'ý': 'ys', 'Ý': 'Ys', 'ỳ': 'yf', 'Ỳ': 'Yf', 'ỷ': 'yr', 'Ỷ': 'Yr', 'ỹ': 'yx', 'Ỹ': 'Yx', 'ỵ': 'yj', 'Ỵ': 'Yj',
    'đ': 'dd', 'Đ': 'Dd',
}


"""
region: Các biến thể của các nguyên âm có dấu.
region2: Các biến thể của phụ âm hoặc cụm phụ âm.
"""
region = {
    'ả': 'ã', 'ã': 'ả', 'ẻ': 'ẽ', 'ẽ': 'ẻ', 'ỉ': 'ĩ', 'ĩ': 'ỉ', 'ỏ': 'õ', 'õ': 'ỏ', 'ở': 'ỡ', 'ỡ': 'ở', 'ổ': 'ỗ', 'ỗ': 'ổ', 'ủ': 'ũ', 'ũ': 'ủ', 'ử': 'Ữ', 'ữ': 'ử', 
    'ỷ': 'ỹ', 'ỹ': 'ỷ', 
    'Ả': 'Ã', 'Ã': 'Ả', 'Ẻ': 'Ẽ', 'Ẽ': 'Ẻ', 'Ỉ': 'Ĩ', 'Ĩ': 'Ỉ', 'Ỏ': 'Õ', 'Õ': 'Ỏ', 'Ở': 'Ỡ', 'Ỡ': 'Ở', 'Ổ': 'Ỗ', 'Ỗ': 'Ổ', 'Ủ': 'Ũ', 'Ũ': 'Ủ', 'Ử': 'Ữ', 'Ữ': 'Ử',
    'Ỷ': 'Ỹ', 'Ỹ': 'Ỷ', 
}

region2 = {
    'ch': 'tr', 'tr': 'ch', 
    'Ch': 'Tr', 'Tr': 'Ch',
    'd': 'gi', 'gi': 'd', 
    'D': 'Gi', 'Gi': 'D',
    'l': 'n', 'n': 'l', 
    'L': 'N', 'N': 'L',
    'x': 's', 's': 'x', 
    'X': 'S', 'S': 'X'
}

"""
Mỗi từ thường được viết tắt thành một chuỗi ký tự ngắn hơn. Ví dụ, 'anh' được viết tắt thành 'a'
"""
acronym = {
    'anh': 'a', 'biết': 'bít', 'chồng': 'ck', 'được': 'dc', 'em': 'e', 'gì': 'j', 'giờ': 'h',
    'Anh': 'A', 'Biết': 'Bít', 'Chồng': 'Ck', 'Được': 'Dc', 'Em': 'E', 'Gì': 'J', 'Giờ': 'H',
    'không': 'ko', 'muốn': 'mún', 'ông': 'ôg', 'phải': 'fai', 'tôi': 't', 'vợ': 'vk', 'yêu': 'iu',
    'Không': 'Ko', 'Muốn': 'Mún', 'Ông': 'Ôg', 'Phải': 'Fai', 'Tôi': 'T', 'Vợ': 'Vk', 'Yêu': 'Iu',
}

"""
Mục đích: Thêm các từ viết tắt vào câu để tăng cường dữ liệu.
Chi tiết:
Tạo một số ngẫu nhiên random từ 0 đến 1.
Nếu random > 0.5, duyệt qua các từ trong acronym và thay thế chúng bằng dạng viết tắt với xác suất 50%.
Nếu random <= 0.5, trả về câu gốc không thay đổi.
"""
def _teen_code(sentence):
    random = np.random.uniform(0,1,1)[0]
    new_sentence = str(sentence)

    if random > 0.5:
        for word in acronym.keys():
            # Tìm và thay thế từ hoặc cụm từ trong câu, không dùng biên giới từ (\b) cho các cụm từ có dấu cách
            if word in new_sentence:
                random2 = np.random.uniform(0,1,1)[0]
                if random2 < 0.5:
                    new_sentence = new_sentence.replace(word, acronym[word])
        return new_sentence
    else:
        return sentence


"""
Mục đích: Thêm nhiễu vào câu để tạo ra dữ liệu tăng cường, giúp mô hình học cách khắc phục các lỗi phổ biến.
Chi tiết:
Gọi hàm _teen_code để có thể thêm từ viết tắt.
Duyệt từng ký tự trong câu:
Nếu ký tự không phải là chữ cái (full_letters), giữ nguyên.
Với xác suất 94%, giữ nguyên ký tự.
Với xác suất 3.5%, thực hiện các thay đổi dựa trên từ điển typo và region.
Với xác suất 5.5%, có thể thay đổi phụ âm dựa trên region2 hoặc đảo chữ cái.
Hàm này sử dụng nhiều lớp ngẫu nhiên để quyết định cách thay đổi ký tự, tạo ra sự đa dạng trong dữ liệu.
"""
def _add_noise(sentence):
    sentence = _teen_code(sentence)
    noisy_sentence = ''
    i = 0

    while i < len(sentence):
        if sentence[i] not in full_letters:
            noisy_sentence += sentence[i]
        else:
            random = np.random.uniform(0,1,1)[0]
            if random <= 0.94:
                noisy_sentence += sentence[i]
            elif random <= 0.985:
                if sentence[i] in typo:
                    if sentence[i] in region:
                        random2 = np.random.uniform(0,1,1)[0]
                        if random2 <= 0.4:
                            noisy_sentence += ''.join(typo.get(sentence[i], [sentence[i]]))
                        elif random2 <= 0.8:
                            noisy_sentence += ''.join(region.get(sentence[i], [sentence[i]]))
                        elif random2 <= 0.95:
                            noisy_sentence += unidecode(sentence[i])
                        else:
                            noisy_sentence += sentence[i]
                    else:
                        noisy_sentence += ''.join(typo.get(sentence[i], [sentence[i]]))
                else:
                    random3 = np.random.uniform(0,1,1)[0]
                    if random3 <= 0.6:
                        noisy_sentence += ''.join(typo.get(sentence[i], [sentence[i]]))
                    elif random3 < 0.9:
                        noisy_sentence += unidecode(sentence[i])
                    else:
                        noisy_sentence += sentence[i]
            elif i == 0 or sentence[i-1] not in full_letters:
                random4 = np.random.uniform(0,1,1)[0]
                if random4 <= 0.9:
                    if i < len(sentence) - 1 and sentence[i] in region2.keys() and sentence[i+1] in vowel:
                        noisy_sentence += region2[sentence[i]]
                    elif i < len(sentence) - 2 and sentence[i:i+2] in region2.keys() and sentence[i+2] in vowel:
                        noisy_sentence += region2[sentence[i:i+2]]
                        i += 1
                    else:
                        noisy_sentence += sentence[i]
                else:
                    noisy_sentence += sentence[i]
            else:
                new_random = np.random.uniform(0, 1)
                if new_random <= 0.33 and i < len(sentence) - 1:
                    noisy_sentence += sentence[i+1]
                    noisy_sentence += sentence[i]
                    i += 1
                else:
                    noisy_sentence += sentence[i]
        i += 1
    return noisy_sentence

"""
Định nghĩa bộ ký tự (alphabet) mà mô hình sẽ sử dụng để mã hóa và giải mã văn bản.
Chi tiết:
'\x00': Thường được sử dụng như một ký tự đặc biệt (padding).
' ' (space): Để phân tách các từ.
Các chữ số từ 0 đến 9.
Các chữ cái tiếng Việt bao gồm cả nguyên âm và phụ âm.
"""
alphabet = ['\x00', ' '] + list('0123456789') + full_letters


"""
Mục đích: Xử lý văn bản để tách thành các cụm từ (phrases) phù hợp để huấn luyện mô hình.
Chi tiết:
Duyệt qua từng câu trong data.
Thay thế hoặc loại bỏ các ký tự không thuộc alphabet:
Sử dụng unidecode để chuyển các ký tự không thuộc alphabet thành dạng không dấu hoặc loại bỏ chúng.
Sử dụng biểu thức chính quy re.findall(r'\w[\w\s]+', text) để tìm các cụm từ chứa các ký tự chữ cái và khoảng trắng.
Loại bỏ các cụm từ có ít hơn 2 từ.
In ra số lượng cụm từ sau khi xử lý.
"""
# phrases = []
# for text in data:
#     # Thay thế hoặc xóa bỏ các ký tự thừa
#     for c in set(text):
#         if re.match('\w', c) and c not in alphabet:
#             uc = unidecode(c)
#             if re.match('\w', uc) and uc not in alphabet:
#                 text = re.sub(c, '', text)
#             else:
#                 text = re.sub(c, uc, text)
#     phrases += re.findall(r'\w[\w\s]+', text)

# phrases = [p.strip() for p in phrases if len(p.split()) > 1]
# print("số đoạn: ",len(phrases))


"""
Mục đích: Tạo các n-gram từ các cụm từ để tăng cường dữ liệu huấn luyện.
Chi tiết:
NGRAM = 5: Số lượng từ trong mỗi n-gram.
MAXLEN = 39: Độ dài tối đa của một n-gram.
Duyệt qua từng cụm từ trong phrases:
Nếu số từ trong cụm từ lớn hơn hoặc bằng NGRAM, tạo tất cả các n-gram có độ dài NGRAM và độ dài không vượt quá MAXLEN.
Nếu số từ nhỏ hơn NGRAM nhưng độ dài không vượt quá MAXLEN, thêm cụm từ đó vào list_ngrams.
Loại bỏ các n-gram trùng lặp bằng cách chuyển list_ngrams thành set rồi lại chuyển ngược thành list.
In ra số lượng n-gram sau khi xử lý.
"""
NGRAM = 5
MAXLEN = 39
# list_ngrams = []
# for p in phrases:
#     list_p = p.split()
#     if len(list_p) >= NGRAM:
#         for ngr in ngrams(p.split(), NGRAM):
#             if len(' '.join(ngr)) <= MAXLEN:
#                 list_ngrams.append(' '.join(ngr))
#     elif len(' '.join(list_p)) <= MAXLEN:
#         list_ngrams.append(' '.join(list_p))

# list_ngrams = list(set(list_ngrams))
# print(len(list_ngrams))

"""
Mục đích: Chuyển đổi văn bản thành định dạng one-hot encoding để sử dụng trong mô hình học máy.
Chi tiết:
Tạo một mảng numpy x có kích thước (MAXLEN, len(alphabet)), khởi tạo bằng 0.
Duyệt qua từng ký tự trong văn bản (tối đa MAXLEN ký tự):
Đánh dấu vị trí của ký tự đó trong alphabet bằng cách đặt giá trị 1 tại vị trí tương ứng.
Nếu số ký tự trong văn bản ít hơn MAXLEN, điền vào các vị trí còn lại bằng cách đặt 1 ở cột đầu tiên ('\x00').
Trả về mảng mã hóa x.
"""
# def _encoder_data(text):
#   x = np.zeros((MAXLEN, len(alphabet)))
#   for i, c in enumerate(text[:MAXLEN]):
#     x[i, alphabet.index(c)] = 1
#   if i <  MAXLEN - 1:
#     for j in range(i+1, MAXLEN):
#       x[j, 0] = 1
#   return x

"""
Mục đích: Chuyển đổi định dạng one-hot encoding trở lại thành văn bản.
Chi tiết:
Tìm chỉ số có giá trị lớn nhất (1) trong mỗi hàng của mảng x bằng argmax.
Chuyển các chỉ số này thành ký tự tương ứng trong alphabet.
Nối các ký tự lại thành một chuỗi văn bản.
"""
def _decoder_data(x):
  x = x.argmax(axis=-1)
  return ''.join(alphabet[i] for i in x)

"""
Xây dựng một mô hình mạng nơ-ron sâu để xử lý chuỗi ký tự và học cách chỉnh sửa văn bản.
Chi tiết:
Encoder:
Sử dụng một lớp LSTM với 256 đơn vị.
input_shape=(MAXLEN, len(alphabet)): Mỗi đầu vào có độ dài MAXLEN và mỗi ký tự được mã hóa bằng one-hot với kích thước len(alphabet).
return_sequences=True: Trả về toàn bộ chuỗi đầu ra để có thể tiếp nối với decoder.
Decoder:
Sử dụng một lớp LSTM hai chiều (Bidirectional) với 256 đơn vị.
dropout=0.2: Áp dụng dropout để giảm overfitting.
return_sequences=True: Trả về toàn bộ chuỗi đầu ra.
Các lớp tiếp theo:
TimeDistributed(Dense(256)): Áp dụng một lớp Dense cho mỗi thời điểm trong chuỗi.
Activation('relu'): Hàm kích hoạt ReLU.
TimeDistributed(Dense(len(alphabet))): Lớp Dense để chuyển đổi lại thành kích thước của alphabet.
Activation('softmax'): Hàm kích hoạt Softmax để xác định xác suất của mỗi ký tự trong alphabet.
Biên dịch mô hình:
Sử dụng hàm mất mát categorical_crossentropy phù hợp với bài toán phân loại đa lớp.
Sử dụng optimizer Adam với tốc độ học 0.001.
Theo dõi chỉ số accuracy.
"""
# encoder = LSTM(256, input_shape=(MAXLEN, len(alphabet)), return_sequences=True)
# decoder = Bidirectional(LSTM(256,  return_sequences=True, dropout=0.2))

# model = Sequential()
# model.add(encoder)
# model.add(decoder)

# model.add(TimeDistributed(Dense(256)))
# model.add(Activation('relu'))
# model.add(TimeDistributed(Dense(len(alphabet))))
# model.add(Activation('softmax'))

# model.compile(loss='categorical_crossentropy',
#               optimizer=Adam(learning_rate=0.001),
#               metrics=['accuracy'])
# model.summary()
# from tensorflow.keras.utils import plot_model
# plot_model(model, to_file='model.png')

"""
Mục đích: Chia dữ liệu thành hai tập: huấn luyện (80%) và kiểm tra (20%) để đánh giá hiệu suất mô hình.
Chi tiết:
Sử dụng train_test_split từ thư viện sklearn để chia dữ liệu.
test_size=0.2: 20% dữ liệu dành cho tập kiểm tra.
random_state=42: Đặt seed để đảm bảo kết quả chia dữ liệu có thể tái lập.
"""
from sklearn.model_selection import train_test_split
# train_data, valid_data = train_test_split(list_ngrams, test_size=0.2, random_state=42)

"""
Mục đích: Tạo các batch dữ liệu để huấn luyện mô hình mà không cần tải toàn bộ dữ liệu vào bộ nhớ cùng một lúc.
Chi tiết:
data: Dữ liệu nguồn (tập huấn luyện hoặc kiểm tra).
batch_size: Kích thước của mỗi batch.
Vòng lặp vô hạn (while True) để liên tục cung cấp dữ liệu cho mô hình.
Trong mỗi batch:
y: Đối tượng mục tiêu (dữ liệu gốc được mã hóa).
x: Đầu vào (dữ liệu đã được thêm nhiễu và mã hóa).
Nếu đạt đến cuối dữ liệu, quay lại đầu danh sách.
"""
# # Chia tách dữ liệu để tránh tràn RAM
# BATCH_SIZE = 512
# EPOCHS = 10
# def _generate_data(data, batch_size):
#     current_index = 0
#     while True:
#         x, y = [], []
#         for i in range(batch_size):
#             y.append(_encoder_data(data[current_index]))
#             x.append(_encoder_data(_add_noise(data[current_index])))
#             current_index += 1
#             if current_index > len(data) - 1:
#                 current_index = 0
#         yield (np.array(x), np.array(y))

"""
Mục đích: Tạo các generators để cung cấp dữ liệu cho quá trình huấn luyện và kiểm tra.
Chi tiết:
BATCH_SIZE = 512: Mỗi batch chứa 512 mẫu dữ liệu.
EPOCHS = 10: Số lần lặp qua toàn bộ dữ liệu huấn luyện.
"""
# train_generator = _generate_data(train_data, batch_size = BATCH_SIZE)
# validation_generator = _generate_data(valid_data, batch_size = BATCH_SIZE)

# train model
# H = model.fit(
#     train_generator, epochs = EPOCHS,
#     steps_per_epoch=len(train_data) // BATCH_SIZE,
#     validation_data=validation_generator,
#     validation_steps=len(valid_data) // BATCH_SIZE
# )
# model.save('./my_model/model_22_9_2024.h5')

# Define the path to the model (adjust this path based on your environment)
model_path = './model_27_9_2024.h5'
model = load_model(model_path)

"""
Mục đích: Đảm bảo rằng các hàm mã hóa và giải mã dữ liệu được định nghĩa lại để sử dụng trong phần dự đoán.
Chi tiết: Các hàm này giống như đã được định nghĩa trước đó để chuẩn hóa dữ liệu đầu vào và giải mã kết quả dự đoán.
"""
NGRAM = 5
MAXLEN = 39
alphabet = ['\x00', ' '] + list('0123456789') + full_letters
def _encoder_data(text):
  x = np.zeros((MAXLEN, len(alphabet)))
  for i, c in enumerate(text[:MAXLEN]):
    x[i, alphabet.index(c)] = 1
  if i <  MAXLEN - 1:
    for j in range(i+1, MAXLEN):
      x[j, 0] = 1
  return x
def _decoder_data(x):
  x = x.argmax(axis=-1)
  return ''.join(alphabet[i] for i in x)



"""
Mục đích: Tạo các n-gram từ câu văn sử dụng thư viện NLTK.
Chi tiết:
Sử dụng nltk.ngrams để tạo các n-gram từ danh sách từ.
Chỉ thêm các n-gram có độ dài không vượt quá maxlen.
Nếu số từ ít hơn n, thêm toàn bộ câu như một n-gram duy nhất.
"""
def _nltk_ngrams(sentence, n, maxlen):
    list_ngrams = []
    list_words = sentence.split()
    num_words = len(list_words)

    if num_words >= n:
        for ngram in nltk.ngrams(list_words, n):
            if len(' '.join(ngram)) <= maxlen:
                list_ngrams.append(ngram)
    else:
        list_ngrams.append(tuple(list_words))
    return list_ngrams


"""
Mục đích: Dự đoán sửa lỗi cho một n-gram cụ thể.
Chi tiết:
Kết hợp các từ trong n-gram thành một chuỗi văn bản.
Mã hóa chuỗi văn bản và đưa vào mô hình để dự đoán.
Giải mã kết quả dự đoán và loại bỏ ký tự padding ('\x00').
"""
def _guess(ngram):
    text = " ".join(ngram)
    preds = model.predict(np.array([_encoder_data(text)]))

    return _decoder_data(preds[0]).strip('\x00')

"""
Mục đích: Thêm lại các dấu câu vào văn bản đã được sửa lỗi.
Chi tiết:
Duyệt qua từng từ trong văn bản gốc text để xác định vị trí các dấu câu ở đầu và cuối từ.
Lưu các dấu câu này vào list_punctuation với chỉ số vị trí từ.
Sau đó, kết hợp các từ đã được sửa lỗi corrected_text với các dấu câu tương ứng từ list_punctuation.
"""
def _add_punctuation(text, corrected_text):
    list_punctuation = {}
    for (i, word) in enumerate(text.split()):
        if word[0] not in alphabet or word[-1] not in alphabet:
            # Dấu ở đầu chữ như " và '
            start_punc = ''
            for c in word:
                if c in alphabet:
                    break
                start_punc += c

            # Dấu ở sau chữ như .?!,;
            end_punc = ''
            for c in word[::-1]:
                if c in alphabet:
                    break
                end_punc += c
            end_punc = end_punc[::-1]

            # Lưu vị trí từ và dấu câu trong từ đó
            list_punctuation[i] = [start_punc, end_punc]

    # Thêm dấu câu vào vị trí các từ đã đánh dấu
    result = ''
    for (i, word) in enumerate(corrected_text.split()):
        if i in list_punctuation:
            result += (list_punctuation[i][0] + word + list_punctuation[i][1]) + ' '
        else:
            result += word + ' '

    return result.strip()

"""
Mục đích: Chỉnh sửa văn bản nhập vào bằng cách sử dụng mô hình đã huấn luyện để sửa lỗi.
Chi tiết:
Bước 1: Loại bỏ các ký tự đặc biệt không thuộc alphabet.
Bước 2: Tạo các n-gram từ văn bản đã được làm sạch.
Bước 3: Dự đoán sửa lỗi cho từng n-gram bằng hàm _guess.
Bước 4: Sử dụng Counter để xác định từ được dự đoán phổ biến nhất tại mỗi vị trí trong câu.
Bước 5: Kết hợp các từ đã được sửa lỗi thành văn bản cuối cùng và thêm lại các dấu câu.
"""
def _correct(text):
    # Xóa các ký tự đặc biệt
    new_text = re.sub(r"[^" + ''.join(alphabet) + ']', '', text)
  

    ngrams = list(_nltk_ngrams(new_text, NGRAM, MAXLEN))
    guessed_ngrams = list(_guess(ngram) for ngram in ngrams)
    candidates = [Counter() for _ in range(len(guessed_ngrams) + NGRAM - 1)]

    for nid, ngram in enumerate(guessed_ngrams):
        for wid, word in enumerate(re.split(r'\s', ngram)):
            index = nid + wid
            # print(f'nid: {nid}, wid: {wid}, index: {index}, candidates length: {len(candidates)}')
            if index < len(candidates):
                candidates[index].update([word])
            else:
                # Safely append to candidates if the index exceeds the current list length
                candidates.append(Counter([word]))
                # print(f"Index {index} is out of range, adding new Counter!")

    corrected_text = ' '.join(c.most_common(1)[0][0] for c in candidates if c)
    return _add_punctuation(text, corrected_text)



# while(True):
#     text = input("nhập nội dung/nhấn 'q' để thoát: ")
#     if text == 'q':
#         break
#     print(_correct(text))

    # result = _correct(text)
    # text = re.sub(r"[^" + ''.join(alphabet) + ']', '', text)
    # list_text = text.split()

    # result = re.sub(r"[^" + ''.join(alphabet) + ']', '', result)
    # list_result = result.split()

    # hien thi cac tu da sua sai
    # Hiển thị những từ đã sửa
    # corrected_word = [(list_text[i], list_result[i]) for i in range(len(list_text)) if list_text[i] != list_result[i]]
    # print(corrected_word)




# chong thowfi đại soos óha hiện lay, văn bản đánh máy ddax daafn thay thế vawn barn viết tay bởi sự thuận tieejn của nó. Kefm theo đó, lỗi chính tả xuất hiện trog lúc soạn tharo nà điều ko thể tránh khỏi. Một số ný do gaay nên lỗi chính tả là: lỗi gõ bàn phím, khác biệt vùng mieefn, viết tắt,.. Điều này đã dấn đến nhu cầu hệ thống giúp phát hiện và sửa lỗi chính tả trong văn bản tiếng Việt.


"""
    sử dụng kiến trúc Seq2Seq với LSTM. Để dễ hiểu, tôi sẽ giải thích các phần chính trong mã và cách mà mô hình này hoạt động:

### 1. Chuẩn bị Dữ liệu:
- **Đọc Dữ liệu**: Đoạn mã bắt đầu bằng việc đọc dữ liệu từ tệp `corpus.txt`, mỗi dòng trong tệp này đại diện cho một câu hoặc đoạn văn bản.
- **Tiền xử lý**: Các nguyên âm và phụ âm tiếng Việt được xác định. Bên cạnh đó, có các quy tắc để ánh xạ các ký tự có dấu thành các ký tự không dấu hoặc từ viết tắt, giúp tăng cường dữ liệu.

### 2. Tạo Dữ liệu Huấn Luyện:
- **Thêm Nhiễu**: Hàm `_add_noise` làm cho dữ liệu huấn luyện phong phú hơn bằng cách thêm lỗi hoặc biến thể vào câu gốc. Điều này giúp mô hình học cách sửa lỗi phổ biến.
- **N-grams**: Các cụm từ (n-grams) được tạo từ dữ liệu văn bản, từ đó giúp mô hình học cách nhận dạng và sửa chữa lỗi trong văn bản.

### 3. Mô Hình Seq2Seq với LSTM:
- **Kiến trúc Mô Hình**: Mô hình sử dụng lớp LSTM (Long Short-Term Memory) cho cả encoder và decoder. LSTM là một loại mạng nơ-ron hồi tiếp rất hiệu quả cho các nhiệm vụ liên quan đến chuỗi (sequence), như dịch máy hoặc sửa lỗi văn bản.
- **Bi-directional LSTM**: Sử dụng lớp LSTM hai chiều, nghĩa là nó xem xét thông tin từ cả hai đầu của chuỗi đầu vào, giúp cải thiện khả năng hiểu ngữ cảnh của mô hình.

### 4. Huấn Luyện Mô Hình:
- **Tạo Generators**: Dữ liệu được cung cấp cho mô hình theo hình thức batches thông qua generator, giúp tiết kiệm bộ nhớ.
- **Huấn luyện**: Mô hình được huấn luyện trên dữ liệu đã qua xử lý, sử dụng hàm mất mát `categorical_crossentropy` và tối ưu hóa bằng cách sử dụng Adam optimizer.

### 5. Dự Đoán và Sửa Lỗi:
- **Dự Đoán**: Sau khi huấn luyện, mô hình có thể dự đoán và sửa lỗi cho các n-grams mới bằng cách mã hóa chúng, chạy qua mô hình và sau đó giải mã kết quả.
- **Thêm Dấu Câu**: Sau khi sửa lỗi, hàm `_add_punctuation` được sử dụng để thêm lại dấu câu vào văn bản đã sửa, đảm bảo văn bản đầu ra có hình thức đúng.

### Tóm tắt:
Đoạn mã này xây dựng một mô hình sửa lỗi văn bản tiếng Việt bằng cách sử dụng kiến trúc Seq2Seq với LSTM. Bằng cách chuẩn bị dữ liệu, thêm nhiễu để tăng cường, và sử dụng kiến trúc nơ-ron hồi tiếp để học từ dữ liệu, mô hình có khả năng dự đoán và sửa chữa các lỗi phổ biến trong văn bản. 

Nếu bạn có bất kỳ câu hỏi cụ thể nào về từng phần của mã hoặc cần giải thích chi tiết hơn về bất kỳ khái niệm nào, hãy cho tôi biết!
"""




# -------------------------------------------------------
# Load environment variables from .env file
app = Flask(__name__)
api_key = "MNhoyUpSllDVcFteZKCsUdK6nOUwh5gUK7cwihj6"



def create_loader():
    file_path = "./luatBaoHiemXaHoi.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text

def faiss_index():
    full_text = create_loader()
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False
    )
     # Split text into chunks
    naive_chunks = text_splitter.split_text(full_text)
    # print(naive_chunks)

     # Create a vector store and retriever
    naive_chunk_vectorstore = FAISS.from_texts(
        naive_chunks, 
        embedding=CohereEmbeddings(cohere_api_key=api_key, model="embed-multilingual-v3.0")
    )    
    naive_chunk_vectorstore.save_local('E:\\CTU\\luanvan\\test_rag_sematic\\docs_index_luat_BHXH')

# Initialize RAG Chain on startup
def initialize_rag_chain():
    global naive_chunk_vectorstore
    os.environ["COHERE_API_KEY"] = api_key

    naive_chunk_vectorstore = FAISS.load_local(
        'E:\\CTU\\luanvan\\test_rag_sematic\\docs_index_luat_BHXH', 
        CohereEmbeddings(cohere_api_key=api_key, model="embed-multilingual-v3.0"),
        allow_dangerous_deserialization=True
    )
    # Define prompt template for retrieval-augmented generation (RAG)
    rag_template = """
    Không trả lời các câu hỏi về chính trị, bạo lực.
    Sử dụng các phần ngữ cảnh sau để trả lời câu hỏi của người dùng. Bạn là một chuyên gia về luật bảo hiểm xã hội Việt Nam. Xin vui lòng cho tôi biết khoản nào điều mấy trong Luật bảo hiểm xã hội Việt Nam tương ứng với nội dung câu hỏi và giải thích đầy đủ theo luật
    User's Query:
    {question}
    
    Context:
    {context}
    """
    rag_prompt = ChatPromptTemplate.from_template(rag_template)
    # Initialize Cohere model
    base_model = ChatCohere(model="command-r-08-2024", temperature=0)

    naive_chunk_retriever = naive_chunk_vectorstore.as_retriever()
    naive_chunk_retriever.search_kwargs['fetch_k'] = 30
    naive_chunk_retriever.search_kwargs['maximal_marginal_relevance'] = True
    naive_chunk_retriever.search_kwargs['k'] = 10

    # Define the retrieval-augmented generation chain
    naive_rag_chain = (
        {"context": naive_chunk_retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | base_model
        | StrOutputParser()
    )

    return naive_rag_chain

# Initialize the RAG chain once when the app starts
print("server khởi động lại")
rag_chain = initialize_rag_chain()


@app.route("/")
def index():
    return render_template("index.html")

# URL API của Rasa (thay đổi địa chỉ IP và cổng nếu khác)
RASA_API_URL = "http://localhost:5055/webhook"


@app.route("/check_spell", methods=["POST"])
def check_spell():
    data = request.get_json()
    user_question = data.get("question")  # Sử dụng trường 'question'
    print(f"Received question: {user_question}")
    if not user_question.strip():
        return jsonify({"error": "Câu hỏi không được để trống."}), 400
    try:
        response = _correct(user_question)
        return jsonify({"response": response})  # Sử dụng trường 'response'
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_question = data.get("question")  # Sử dụng trường 'question'
    print(f"Received question: {user_question}")
    if not user_question.strip():
        return jsonify({"error": "Câu hỏi không được để trống."}), 400
    try:
        # response = _correct(user_question)
        # Invoke the RAG chain với câu hỏi của người dùng
        response = rag_chain.invoke(user_question)
        return jsonify({"response": response})  # Sử dụng trường 'response'
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Chạy code này khi thêm dữ liệu mới vào txt
    # faiss_index()
    app.run(host='0.0.0.0', port=5000)