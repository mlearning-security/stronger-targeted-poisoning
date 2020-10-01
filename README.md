This is a Python implementaion of "Stronger Targeted Poisoning Attacks Against Malware Detection".

The datasets used in our validation (Ransomware and M-EMBER) are not included in the repository. 





#### ARGUMENTS

* -d [dataset]     :[dataset] is ransom or ember

* -t [int]         :steps for gradient discent in poisoning 

* -selection [int] :#features

* -epo [int]       :#learning steps for detection model

* -shuffle         :shuffle D_train U D_p before training

* -term            :use ε in Back-gradient algorithm

* -eps [float]     :ε for Back-gradient algorithm

* -eta [float]     :learning rate η for Back-gradient algorithm

* -decay           :diminish η

* -max [int]       :#iterations for Back-gradient algorithm

* -phi             :generated poisoning data satisfies the value range（ransom:{0,1}, ember:[0,1])

* -p [int]         :#poison

* -gpu [int]       :0:use gpu, -1:use cpu

* -multi           :generate {targeted} poisoning data

* -mulmode 1       :initial poisoning data is chosen from targeted malware (and label is flipped)

* -d_seed [int]    :seed value for random selection of data to be used from the whole data set

* -id [int]        :malware family ID for targeted malware

* -save [str]      :output directory

* -constraint      :1: use constraint term (Sasaki's extension)

* -beta [float]    :coefficient for the constraint term 

* -sphere          :generate poisoning data in F_good

* -elim [float]    :outlier removal ratio (default:0.15)

* -flip            :label flip attack

* -solver          :attack using solver

ID1→-d ransom -id 1
ID2→-d ransom -id 5
ID3→-d ember -id 1
ID4→-d ember -id 6
ID5→-d ember -id 9



#### EXAMPLE

* basic attack for ID2 : 

  ```python 02_ransom.py -d ransom -id 5 -t 200  -selection 400 -epo 10000 -shuffle -term -eps 1e-4 -eta 0.3 -decay -max 10 -phi -p 5 -gpu -1 -multi -mulmode 1 -sphere```

  

* solver for ID1 : 

  ```python 02_ransom.py -d ransom -t 200 -selection 400 -epo 10000 -shuffle -term -eps 1e-4 -eta 0.3 -decay -max 10 -phi -p 5 -gpu -1 -multi -mulmode 1 -id 1 -d_seed 0 -sphere -solver```

* solver for ID4 : 

   ```python 01_ember.py -d ember -p 10 -id 6 -d_seed 10 -scaler 0 -t 100 -epo 2000 -shuffle -term -eps 1e-4 -eta 0.3 -decay -max 100 -phi  -gpu -1 -multi -mulmode 1```

