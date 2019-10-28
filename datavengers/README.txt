####### HOW TO USE PERSONALITY PREDICTOR PERSONALITY.PY

### 1) TO INSTANTIATE THE CLASS
Run "model = Personality()"
At this point, the model does not exist yet

### 2) TO CREATE A MODEL AND TRAIN IT
Run "model.train(your_data)"
At this point, a new model had been created and trained 

### 3) TO PREDICT NEW DATA'S OUTPUT
Run "model.predict(your_data)"
Please know that this method should not be run before prior call to "train" or "load"  

### 4) TO LOAD A MODEL FROM EXISTING FILE 
Run "model.load()"
This method loads an existing trained model from 'persistence/personality/personality.model'
Please know that this replaces the exiting model by this one.

### 5) TO SAVE A MODEL 
Run "model.save()"
This method saves the current model under 'persistence/personality/personality.model'

