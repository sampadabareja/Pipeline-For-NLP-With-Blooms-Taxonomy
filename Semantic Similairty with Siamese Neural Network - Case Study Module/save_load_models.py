from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import model_from_json

def save_current_model(model):

  print("Saving Models ..")
  # Save the trained weights of the model :
  model.save_weights('Models/100D  Adam/model_weights.h5')

  # Save the current  model architecture :
  with open('Models/100D  Adam/model_architecture.json', 'w') as f:
    f.write(model.to_json())

  # Save the tokenizer iin JSON format for later use  :
  with open('Models/100D  Adam/tokenizer.json', 'w') as f:
    f.write(tokenizer.to_json())
       
  print("Models Saved Successfully With Tokenizer!")

  

def load_mods():

  print("Loading Models ..")

  with open('/content/drive/MyDrive/Siamese/Models/100D  Adam/tokenizer.json') as f:
    tokenizer = tokenizer_from_json(f.read())

    # Model from JSON :
  with open('/content/drive/MyDrive/Siamese/Models/100D  Adam/model_architecture.json', 'r') as f:
    model = model_from_json(f.read())

     # Load weights :
  model.load_weights('/content/drive/MyDrive/Siamese/Models/100D  Adam/model_weights.h5')
    
  print("Loaded Models Successfully !")

  return model, tokenizer


 

