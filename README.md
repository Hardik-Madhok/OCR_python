# OCR_python
 This repository contains the python code for Handwritten Greek Letter detection
 This project was more of a self given task. 
 This task is divided into various steps. Some steps are added intentionally just to make the concept more understandable. So, it can be said that this model can be more memory efficient than the present model.
 The dataset is made by my ownself as shown in clipping directory. Clipping code was used to clip letter from original dataset.
 Once dataset was prepared our next step was to remove the noise.
 Here noise is divided into two parts one is backgroud of paper i.e. table, notebook etc and second is the shadow on the paper
 Both the noises are removed step by step starting from background of paper (clearing_background.py) and then removing shadows (clearing_shadows.py)
 As this is supervised learning I manually deleted some images after removing dataset.
 I also took one image from each class before training just to test my model. Therefore, training does not include those images which are used for testing purpose.
 train_1.py is the basic and less complex model which is used for training.
 test_1.py is the code for testing the same.
 I hope this will be helpful in many cases.
 I am open for suggestions and new ideas
 Thank you
