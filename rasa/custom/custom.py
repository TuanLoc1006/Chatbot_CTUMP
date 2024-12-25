import pandas as pd 
stop_words = pd.read_csv('C:\\xampp\htdocs\\chatbot\\rasa\\custom\\vietnamese_stopwords.txt')
print(stop_words)

class Custom:

    def remove_stop_words(chuoi):
        results = []
        for text in chuoi:
            tmp = text.split(' ')
            # Loại bỏ các từ có trong stop_words
            filtered_words = [word for word in tmp if word not in stop_words]
            results.append(" ".join(filtered_words))
        return results

    def check_key(dic, key):
        if key in dic:
            return True
        for key, value in dic.items():
            if isinstance(value, dict):
                if check_key(value, key):
                    return True
        return False