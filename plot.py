import pandas as pd
import matplotlib.pyplot as plt



train_log = r'output\training_log.csv'
df = pd.read_csv(train_log)


epochs = df['epoch']
train_acc = df['training_acc']
val_acc =  df['val_acc']
train_loss = df['training_loss']
val_loss = df['val_loss']

plt.plot(epochs, train_acc, 'b', label='Training acc', marker=">")
plt.plot(epochs, val_acc, 'r', label='Validation acc', marker="d")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, train_loss, 'b', label='Training loss', marker=">")
plt.plot(epochs, val_loss, 'r', label='Validation loss', marker="d")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()