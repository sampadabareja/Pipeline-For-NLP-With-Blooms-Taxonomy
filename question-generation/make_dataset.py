import numpy as np  
import pandas as pd 
import json         



train_input_file_path = r"Enter path to train data"  # Json File
dev_input_file_path   = r"Enter path to dev data"   # Json File

def make_dataframe(file):  

    f = open ( file , "r") 
    data = json.loads(f.read())               
    iid = []                                  
    tit = []                                  
    con = []
    Que = []
    Ans_st = []
    Txt = []
    
    for i in range(len(data['data'])):      
        
        title = data['data'][i]['title']

        for p in range(len(data['data'][i]['paragraphs'])):  
            
            context = data['data'][i]['paragraphs'][p]['context']

            for q in range(len(data['data'][i]['paragraphs'][p]['qas'])): 
                
                question = data['data'][i]['paragraphs'][p]['qas'][q]['question']

                Id = data['data'][i]['paragraphs'][p]['qas'][q]['id']

                for a in range(len(data['data'][i]['paragraphs'][p]['qas'][q]['answers'])): 
                    
                    ans_start = data['data'][i]['paragraphs'][p]['qas'][q]['answers'][a]['answer_start']

                    text = data['data'][i]['paragraphs'][p]['qas'][q]['answers'][a]['text']
                    
                    tit.append(title)
                    con.append(context)
                    Que.append(question)                    
                    iid.append(Id)
                    Ans_st.append(ans_start)
                    Txt.append(text)

    print('Done')      

    new_df = pd.DataFrame(columns=['Id','title','context','question','ans_start','text']) 
    new_df.Id = iid
    new_df.title = tit           
    new_df.context = con
    new_df.question = Que
    new_df.ans_start = Ans_st
    new_df.text = Txt

    final_df = new_df.drop_duplicates(keep='first')  

    return final_df

train_csv = make_dataframe(train_input_file_path)
test_csv  = make_dataframe(dev_input_file_path)

train_csv.to_csv("data/train.csv" , index = False)  # Save the final train csv file 
test_csv.to_csv("data/test.csv" , index = False)    # Save the final test csv file