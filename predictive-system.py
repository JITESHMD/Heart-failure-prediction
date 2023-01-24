
import numpy as np
import pickle

loaded_model=pickle.load(open("C:/Users/mdjit/OneDrive/Desktop/devlop/Heart-Failure-Prediction/trained_model.sav",'rb'))


input_data=(65,0,1,149,341,1,2,125,0,2.5,2)
input_arr=np.array(input_data)
inputs_array=input_arr.reshape(1,-1)
inputs_array
if (loaded_model.predict(inputs_array))==1:
    print("YOUR HEART HAS GOT FAILURE")
else:
    print("YOUR HEART IS NOT FAILURED")



