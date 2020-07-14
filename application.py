#!/usr/bin/env python
# coding: utf-8

# # Predict the input data 

# # Flask Application

##################################################
# Library
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import pickle
import psycopg2,os,csv

from keras.models import Sequential,load_model
from keras.layers import LSTM, Dense, Dropout,Embedding
from keras.utils import np_utils,plot_model

from xml.etree import ElementTree
from lxml import etree

from pickle import dump

from IPython.display import display
pd.options.display.max_columns = None
pd.options.display.max_rows =100
pd.set_option('display.max_colwidth', -1)


def pickle_(type_):
    # Loading the model and dictionary of characters and interger sequence
    global model,char_to_integer,integer_to_char,chars,len_input_NN,cat_to_integer,integer_to_cat
    pickle_name = 'Models/ML_LCT_'+type_+'_dict.pkl'
    pickle_in = open(pickle_name,"rb")
    pickle_ip_dict = pickle.load(pickle_in)
    pickle_in.close()

    model_file = 'Models/ML_LCT_'+type_+'_Model.h5'
    model = load_model(model_file)

# # Assigning the variables according to trained model
    char_to_integer  = pickle_ip_dict['char_to_integer']
    integer_to_char  = pickle_ip_dict['integer_to_char']
    chars            = pickle_ip_dict['chars']
    len_input_NN     = pickle_ip_dict['len_input_NN']
    cat_to_integer   = pickle_ip_dict['cat_to_integer']
    integer_to_cat   = pickle_ip_dict['integer_to_cat']


# below def generates the values in to a sequence which is understandable by neural network
def generate_input_sequence(data_):
    data_chr_int =[0]*len_input_NN
    i_ = 0 
    for char in data_:
#         data_chr_int[i_] = char_to_integer[char]
        try:
            data_chr_int[i_] = char_to_integer[char]
        except:
            data_chr_int[i_] = 0
        if i_ >= len_input_NN-1:
            break
        i_+=1
    data_chr_int = np.reshape(data_chr_int, (1, len_input_NN, 1)) 
    data_chr_int = data_chr_int/float(len(chars))
    return data_chr_int

# below def gives the confidence level of each prediction
def confident_score(y_pred):
    data = [['test1','test1','test1']]
    dataframe_ = pd.DataFrame([],columns=['index','value','column'])
    dataframe_
    for index,i in enumerate(y_pred[0],start=0):
        if index == 54:
            break
        data = {'index':index,'value':round(i*100,2),'column':integer_to_cat[index]}
        dataframe_ = dataframe_.append(data,ignore_index=True)    

    dataframe_ = dataframe_.sort_values(by=['value'],ascending=False)
    return dataframe_

# Predicting the input_sequence value to a index
def predict_(input_sequence):
    X_input = np.reshape(input_sequence, (1, len(input_sequence), 1))
    y_pred = model.predict(X_input, verbose=0)
    index = np.argmax(y_pred)
#     print(index,integer_to_cat[index])
    return y_pred,index,integer_to_cat[index]


# Processing XML file to predic the values and give the confidence score to each value

def etree_iter_path(node, tag=None, path='.'):
    if tag == "*":
        tag = None
    if tag is None or node.tag == tag:
        yield node, path
    for child in node:
        _child_path = '%s/%s' % (path, child.tag)
        for child, child_path in etree_iter_path(child, tag, path=_child_path):
            yield child, child_path

def tree_struct_orginal(file): 
       
    tree = ElementTree.parse(file)
    root = tree.getroot()    

    orginal_path_ = []
    xmldoc = ElementTree.parse(file)
    for elem, path in etree_iter_path(xmldoc.getroot()):
        orginal_path_.append(path)

    key_value_ = []
    for i in root.iter():
        key_value_.append([i.tag,i.text])

    final_ =[]
    for index,value in enumerate(orginal_path_):
        final_.append([value,key_value_[index][1]])       

    result_ = []
    for i in final_:
        if type(i[1]) != type(None) and '\n\t' not in i[1]:
            result_.append(i) 
    result_values_ = []
    for i in result_:
        result_values_.append(i[1])    
       
    return result_values_,result_ 

def csv_to_DB(df):
    test_df  = pd.DataFrame()
    test_df1 = pd.DataFrame()

    test_df['data']   = df[df.columns[0]].replace("'","").astype(str)
    test_df['result'] = df.columns[0] #[test.columns[0],test.columns[0]] 

    for i in df.columns[1:]:
        test_df1['data'] = df[i].astype(str).str.replace("'","")
        test_df = test_df.append(test_df1,ignore_index=True)
        test_df.result.fillna(i.replace(" ", "_"),inplace=True)
    test_df = test_df[test_df.data!='nan']
    test_df = test_df.rename(columns={'result': 0,'data':1})
    return test_df

def get_xml(file_name):
    file_name = 'TestCase/'+file_name

    df = pd.DataFrame(tree_struct_orginal(file_name)[1])
    df =df[~df[0].str.contains("{")]
    df['Predicted_Values']=''
    df['Confidence_Score']=''

    for index, row in df.iterrows():
        row['Predicted_Values'] = predict_(generate_input_sequence(row[1])[0])[2] # put the predicted interval value 
        row['Confidence_Score'] = confident_score(predict_(generate_input_sequence(row[1])[0])[0]).reset_index()['value'][0]# put that logic to give confidence interval
    
    list_ = df.Predicted_Values.unique()

    df1 = pd.DataFrame()
    df1 = pd.DataFrame(columns=[0,1,'Predicted_Values','Confidence_Score'])
    
    for i in list_:
        df1 = df1.append(df[df['Predicted_Values']==i].sort_values('Confidence_Score',ascending=False))
    return df1

def get_csv(file_name):
    file_name = 'TestCase/'+file_name

    df = csv_to_DB(pd.read_csv(file_name))

    df =df[~df[0].str.contains("{")]
    df['Predicted_Values']=''
    df['Confidence_Score']=''

    for index, row in df.iterrows():
        row['Predicted_Values'] = predict_(generate_input_sequence(row[1])[0])[2] # put the predicted interval value 
        row['Confidence_Score'] = confident_score(predict_(generate_input_sequence(row[1])[0])[0]).reset_index()['value'][0]# put that logic to give confidence interval
    
    list_ = df.Predicted_Values.unique()

    df1 = pd.DataFrame()
    df1 = pd.DataFrame(columns=[0,1,'Predicted_Values','Confidence_Score'])
    
    for i in list_:
        df1 = df1.append(df[df['Predicted_Values']==i].sort_values('Confidence_Score',ascending=False))
    return df1


from flask import Flask, request, render_template, session, redirect
### working code -- which displays the DataFrame

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('Predict/ONEai.html')

@app.route('/', methods=("POST", "GET"))
def html_table():
    file_name = request.form['myfile']
    type_ = request.form['type_']
    save_ = request.form['save_']
    print(file_name,type_,save_)
    pickle_(type_)
    if '.xml' in file_name:
        data_frame = get_xml(file_name)
        data_frame = data_frame.rename(columns={0: 'X_Path',1:'Value'})
        if save_ == 'save':
            data_frame.to_csv('Saved_Mapping/'+type_+'.csv')
        return render_template('Predict/Predict_Confidence.html',  tables=[data_frame.to_html(classes='data')], tables1=[data_frame.groupby(["Predicted_Values"]).first().to_html(classes='data')])
    elif '.csv' in file_name:
        data_frame = get_csv(file_name)
        data_frame = data_frame.rename(columns={0: 'Column',1:'Value'})
        if save_ == 'save':
            data_frame.to_csv('Saved_Mapping/'+type_+'.csv')
        return render_template('Predict/Predict_Confidence.html',  tables=[data_frame.to_html(classes='data')], tables1=[data_frame.groupby(["Predicted_Values"]).first().to_html(classes='data')])

@app.route('/value')
def my_form1():
    return render_template('Predict/value.html')

@app.route('/value', methods=("POST", "GET"))
def html_table1():
    Input = request.form['value_']
    print("Input :" +Input)
    input_sequence = generate_input_sequence(Input)[0]
    # print(input_sequence)
    y_pred = predict_(input_sequence)
    data_frame = confident_score(y_pred[0])
    return render_template('Predict/value_confidence.html',  tables=[data_frame.to_html(classes='data')])

@app.route('/data')
def my_form2():
    return render_template('DataStructure/DataStructure_Home.html')

@app.route('/data', methods=("POST", "GET"))
def html_table2():
    filename = request.form['myfile']
#     print(filename)
    filename = 'TestCase/'+filename
    print(filename)
    
    if '.xml' in filename:
        print("ener")
        df = pd.DataFrame(tree_struct_orginal(filename)[1])
        df =df[~df[0].str.contains("{")]
        df = df.rename(columns={0: 'X_Path',1:'Value'})
        return render_template('DataStructure/Result.html',  tables=[df.to_html(classes='data')], titles=df.columns.values)
    elif '.csv' in filename:
        df = csv_to_DB(pd.read_csv(filename))
        df = df.rename(columns={0: 'Column',1:'Value'})
        return render_template('DataStructure/Result.html',  tables=[df.to_html(classes='data')], titles=df.columns.values)

        
if __name__ == '__main__':
    app.run()
