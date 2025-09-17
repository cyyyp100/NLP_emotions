import numpy as np
import pandas as pd
import sklearn
from IPython.display import display
import tensorflow as tf
tf.__version__
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

df=pd.read_csv('text.csv')
df2=df.copy()
df2=df2.head(1000)

df2['text'] = df2['text'].str.split(' ',)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(lowercase=False, tokenizer=lambda x: x, preprocessor=lambda x: x)

X = vectorizer.fit_transform(df2['text'])

df_encoded = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

print(df_encoded.head(5))

df_concatenated = pd.concat([df_encoded, df2['label']], axis=1)
print(df_concatenated)
print(df_concatenated['label'].loc[df_concatenated['label']>=5])



#(train_data, train_labels), (test_data, test_labels) = df_concatenated.load_data()
#print(train_data)
#print(test_labels)

#train_labels_one_hot = to_categorical(train_labels, num_classes=6)
#test_labels_one_hot = to_categorical(test_labels, num_classes=6)

#train_data = df_encoded.head(800)
#train_labels = df2['label'].head(800)
#test_data = df_encoded[800:999]
#test_labels = df2['label'][800:999]

#train_data_np = train_data.to_numpy().astype('float32')
#train_labels_np = to_categorical(train_labels.to_numpy())
#train_labels_np = tf.keras.utils.to_categorical(train_labels_np, num_classes=6)
#test_data_np = test_data.to_numpy().astype('float32')
#test_labels_np = to_categorical(test_labels.to_numpy())
#test_labels_np = tf.keras.utils.to_categorical(test_labels_np, num_classes=6)

X_test = df_concatenated.head(800).iloc[:, :-1].values
y_test = df_concatenated.head(800).iloc[:, -1].values
X_train = df_concatenated[800:].iloc[:, :-1].values
y_train = df_concatenated[800:].iloc[:, -1].values

y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)



model = Sequential([
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(6, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train_one_hot, epochs=5, batch_size=32)


test_loss, test_acc = model.evaluate(X_test, y_test_one_hot, batch_size=32)

print(f'Test accuracy: {test_acc}')

model.save('modele_trained.keras')

modele_charge = load_model('modele_trained.keras')

def transf(texte,vectorizer):
    texte_preprocessed = [texte]  # Appliquez votre logique de tokenisation/prétraitement si nécessaire
    texte_vect = vectorizer.transform(texte_preprocessed)
    return texte_vect.toarray()

prediction = modele_charge.predict(transf('love',vectorizer))
print(prediction)
classe_predite = np.argmax(prediction, axis=1)
print(classe_predite)




