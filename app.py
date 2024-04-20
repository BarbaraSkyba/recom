from flask import Flask, render_template, request, jsonify, send_file, flash
from flask_socketio import SocketIO, send # pip install flask-socketio
import requests
import re
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import os
import mysql.connector # pip install mysql-connector-python
from mysql.connector import Error
from mysql.connector import errorcode
import pandas as pd
import tiktoken

import openai #pip install openai==0.28.1
import time
import numpy as np
from flask_cors import CORS
import json
import csv
import textract
import time
import pymorphy3 # pip install pymorphy3 pip install -U pymorphy3-dicts-uk
import locale
import re

import deepl

from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
# pip install plotly
# pip install pandas==1.4.1
# pip install pipreqs
import ssl
from math import dist
from langdetect import detect
#import chardet
import datetime

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')#SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')#SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')  # multi-language model

#import translators as ts
#from googletrans import Translator

#print(googletrans.LANGUAGES)

morph = pymorphy3.MorphAnalyzer(lang='uk')

f_cfg = open(".cfg")
f_prepos = pd.read_csv('./dicts/prepos_uk.csv', header=None)
# Convert the DataFrame to a Dictionary
prepos_dict_0 = f_prepos.to_dict(orient='records')
prepos_dict = {}
for i, line in enumerate(prepos_dict_0): 
    #print(line.get(0))
    prepos_dict.update({line.get(0) : line.get(0)})

#prepos_dict = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)}
#print(prepos_dict)

# returns JSON object as 
# a dictionary
cfg = json.load(f_cfg)

apik = cfg["apik"] 
deepl_key = cfg["deepl"] 
api_token = cfg["api_token"] 

translator = deepl.Translator(deepl_key)
"""
try:
    from mysql.connector.connection_cext import CMySQLConnection as MySQLConnection
except ImportError:
    from mysql.connector.connection import MySQLConnection

connection = MySQLConnection()
connection._ssl['version'] = ssl.PROTOCOL_TLSv1_2

connection = mysql.connector.connect( user='u_reslyete', password = 'htmTmsLT' , host = "127.0.0.1" , database = 'reslyete')
cursor = connection.cursor()
cursor.execute('set GLOBAL max_allowed_packet=67108864')
cursor.execute('set GLOBAL net_read_timeout=3600')
cursor.close()
connection.close()
"""
#forge_sql = """ select id, guid, name from	goods where name like '%миття%' """
forge_sql = """ select id, guid, name from	goods """

app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

SPEC_KOEF = 0.5
LEAVES_KOEF = 0.8
reInit = False

CORS(app)

#socketio = SocketIO(app, cors_allowed_origins="*")

def handle_message(message):
    print("Received message: " + message)
    if message != "User connected!":
        send(message, broadcast=False)

HTTP_URL_PATTERN = r'^http[s]*://.+'
# Define root domain to crawl
domain = "kudev.dev"
full_url = "https://help.kvsaas.com/ru/Plan" # "https://help.kvsaas.com/ru/User_Guide" #
full_url_ku = "https://kudev.dev/" # "https://help.kvsaas.com/ru/User_Guide" #

#full_url = "https://help.kvsaas.com/ru/User_Guide/Process_management"
#domain = "openai.com"
#full_url = "https://openai.com/"

# Global variables
urls = ""
urlsindb = set()
df = pd.DataFrame()
df_embed = pd.DataFrame()
df_merged = pd.to_datetime

tokenizer = tiktoken.get_encoding("cl100k_base")

dict_synonym_terms = {}
dict_synonym_values = {}
dict_synonym_values_adds = {}
dict_synonym_values_sorted = {}

dict_dont_change = {}#{"слон":''} # , "пластикові": '', "пластикові": '', "пластикова": '', "пластиковий": ''

def recalc_dist(order_line, answer_line):
  answer_len = len(answer_line.split())
  koeffs = []
  koeff = 0
  koeff_sum = 0

  for key, word in enumerate(answer_line.split()):
    koeffs.append((1/(key+1))/float(answer_len))
    koeff_sum += (1/(key+1))/float(answer_len)

  if koeff_sum > 0:
    for i, k in enumerate(koeffs):
      koeffs[i] = koeffs[i] / koeff_sum

  for key, word in enumerate(order_line.split()):
       for key2, word_res in enumerate(answer_line.split()):
          if (word_res == word): # (word.isnumeric() and (word_res.find(word) >= 0)) or 
              #print(word, word_res, koeffs[key2])
              koeff += koeffs[key2]

  return koeff 

def recalc_dist2(order_line, answer_line, all=False, syn_qtys = {"nums": '0.123455432'}):
  answer_len = len(answer_line.split())
  koeffs = []
  koeff = 1.00
  koeff_sum = 0.00
  order_line_normal = ""
  answer_line_normal = ""
  dist_hist = []
  not_found_words = []
  #normalizing
  order_line = parse_units(order_line)
  answer_line = parse_units(answer_line)
  
  #print('NORMAL0', order_line_normal, answer_line_normal)
#  print('parse', order_line, answer_line)
  for key, word in enumerate(order_line.split()):
    order_line_normal+= morph.parse(word)[0].normal_form + ' '

  for key, word in enumerate(answer_line.split()):
    answer_line_normal+= morph.parse(word)[0].normal_form + ' '

  order_line_normal = replace_with_synonym(order_line_normal)[0]
  answer_line_normal = replace_with_synonym(answer_line_normal)[0]

  for key, word in enumerate(answer_line_normal.split()):
    if not word in prepos_dict.values():
        koeffs.append((1.00 - (1/(key+1)) + 0.1) * 1.00)
        koeff_sum += ((1.00 - 1/(key+1) + 0.1) * 1.00)
    else:
        koeffs.append(1)

  #print('KOEFF', koeffs, order_line_normal, answer_line_normal)
  #print('NORMAL', order_line_normal, answer_line_normal)
  for key, word in enumerate(order_line_normal.split()):
       word_found = False
       for key2, word_res in enumerate(answer_line_normal.split()):
          #if (word_res.find(word) >= 0):
          #print(word, word_res)
#          print('sys_qtys:', word, syn_qtys["nums"].split(','), str(word) in syn_qtys["nums"].split(','))
          if ((word == word_res and len(word) >= 2 and not word.isnumeric()) or ((word == word_res and len(word) >= 2) and word.isnumeric()) ) and not word in prepos_dict.values():
#              print(word, word_res, koeffs[key2], koeff, key2)
              koeff *= koeffs[key2]
              dist_hist.append({word : koeffs[key2]})
              word_found = True

          
       if not word_found and not word.isnumeric() and not word in prepos_dict.values() and all == True:
            #print(word, answer_line_normal.split())
            not_found_words.append(word)

  if  syn_qtys["nums"] != '0.123455432' and len(dist_hist) > 0:
    for key2, word_res in enumerate(answer_line_normal.split()):
         if word_res.isnumeric() and str(word_res) in syn_qtys["nums"].split(','):
            koeff *= koeffs[key2]
            dist_hist.append({word_res : koeffs[key2]})    

  return [koeff, dist_hist, not_found_words] 

#recalc_dist2('Папір А4 чорний', 'Папір копіюв. A4 чорний 100арк. 16г/м2 Axent 3301-01-А')

#recalc_dist('Папір А4 чорний', 'Папір копіюв. A4 чорний 100арк. 16г/м2 Axent 3301-01-А')

def translate_text(text, target_language): 
    response = openai.Completion.create( 
    
    engine="gpt-3.5-turbo-instruct", 
    prompt=f"Translate precisely to {target_language}: {text}", 
    max_tokens=3000, 
    n=1, 
    stop=None, 
    temperature=0.5) 
    #print(response.choices[0].text.strip())
    return response.choices[0].text.strip()

def translate_deepl(text):
    result = translator.translate_text(text, source_lang="RU", target_lang="UK")
    #print(result.text)  # "Bonjour, le monde !"
    return result.text

def translate_text_by_word(text): 
    """
    response = openai.Completion.create( 
            engine="gpt-3.5-turbo-instruct", 
            prompt=f"Translate precisely to {target_language}: {text}", 
            max_tokens=3000, 
            n=1, 
            stop=None, 
            temperature=0.5) 
    """ 
    str = ''       
    for word in text.split():
        print(word, detect(word))
        if not word.isnumeric() and detect(word) == 'ru':
            response = openai.Completion.create( 
                engine="gpt-3.5-turbo-instruct", 
                prompt=f"Translate from Russian to Ukrainian: {word}", 
                max_tokens=3000, 
                n=1, 
                stop=None, 
                temperature=0.7)
            str+=response.choices[0].text.strip() + ' '
            #print(word, detect(word), morph.parse(word)[0].normal_form)
        else:
            str+=word
    #print(response.choices[0].text.strip())
    return str.strip() #response.choices[0].text.strip()    
"""
def translate_text_by_deepl(text): 
    print(text)
    return ts.translate_text(query_text = text, translator = 'deepl', from_language = 'ru', to_language = 'uk')
    """
#def translate_google_trans(text):
#    return Translator.translate(text, src='ru', dest='uk').text

def find_art(df, string):
    #g840926
    newstring = []
    for key, word in enumerate(string.split()):
        #print(word, word.isnumeric(), len(word))
        if word.isnumeric() and len(word) == 6:
            #print('g'+word)
            newstring = df.loc[df['art'] == ('g'+word), ['art', 'text', 'qty_leaves', 'price_spec']]

    return newstring

def ask_for_help(text):

    pass

def parse_units(str):
  res = ''
  for word in str.split():
    sub_word = word
    firstIsNum = False
    for k, l in enumerate(word):
      middleNotIsNum = False
      if l.isnumeric() and k==0:
        firstIsNum = True
      #print(word[k:], word[k:].isnumeric(), firstIsNum)  
      if firstIsNum == True and not l.isnumeric() and not word[k:].isnumeric():
        sub_word = word[0: k] + ' ' + word[k:]
        break
        #print('num', k, l)
    #print(word)
    res +=sub_word + ' '
  return res.strip()

def replace_with_synonym(order_line, showPrint = False):
    order_line_normal = ''
    syn_arr = {}

    order_line_normal_ = order_line.lower()

  # normalize key line
    k = list(dict_synonym_values.items())
    k.sort(key=lambda x:len(x[0]),reverse=True)
  
    #print('K:', k)

    #for i in k :
    #    dict_synonym_values_sorted.update({i[0]:i[1]})
    syn_arr.update({"nums": '0.123455432'})
    #key_val_normal = ''   
    for i in k:#key_val in dict_synonym_values_sorted.keys():
        key_val = i[0]
        key_val_normal = ''
        #for key, word in enumerate(key_val.split()):
        #    key_val_normal += morph.parse(word)[0].normal_form + ' '   
        key_val_normal = key_val #key_val_normal.strip()
        #print('repl4:', key_val_normal, order_line_normal_)
        if  key_val_normal in order_line_normal_:
            #print('replace:', order_line_normal_, key_val, key_val_normal, dict_synonym_values_sorted.get(key_val))
            for k, l in enumerate(order_line_normal_.strip()):
                if showPrint:
                    print(l, key_val_normal[0], order_line_normal_[k:len(key_val_normal)], key_val_normal)
                if l == key_val_normal[0] and order_line_normal_[k:len(key_val_normal)] == key_val_normal:
                    order_line_normal_ = order_line_normal_[0:k] + i[1] + order_line_normal_[k+len(key_val_normal):]
            """
            for word in order_line_normal_.strip().split():
                if word == key_val_normal:
                    order_line_normal_ = order_line_normal_.replace(key_val_normal, i[1])#dict_synonym_values_sorted.get(key_val))
                    if dict_synonym_values_adds.get(key_val)!= None:
                        syn_arr.update({"nums": dict_synonym_values_adds.get(key_val)})
                else:
                    order_line_normal_ = order_line_normal_  
            """

    order_line = order_line_normal_  
    #print('NORMAL:', order_line)

    for key, word in enumerate(order_line.split()):
        word = word.lower().strip()
            
        if word in dict_synonym_values.keys() or word in dict_dont_change.keys():
            norm = word
        else:
            norm = morph.parse(word)[0].normal_form  

    #    print('NORM:', word, norm)
    #print('repl:', order_line, morph.parse(word)[0].normal_form, norm, dict_synonym_values.keys() )
        #print('repl3:', word, dict_dont_change.keys())
        if not norm in dict_synonym_values.keys() or word in dict_dont_change.keys():
            order_line_normal+= word + ' ' #morph.parse(word)[0].normal_form + ' '   
        else:
            order_line_normal+= dict_synonym_values.get(norm) + ' '     
            #syn_arr.append({dict_synonym_values.get(norm): dict_synonym_values_adds.get(norm)})

    #print('repl2:', order_line_normal, syn_arr)
    return [order_line_normal.strip(), syn_arr]

def parse_lines_synonym(Lines):
    global dict_synonym_terms
    global dict_synonym_values
    global dict_synonym_values_adds
    global dict_synonym_values_sorted
    dict = []
    i = 0
    count = 0
#    print(Lines)
    for line in Lines.index:
        #print('line', line)
        #print(Lines[0][line], Lines[1][line], Lines[2][line], Lines[3][line])
        #dict_synonym_termins = {}
        #dict_synonym_values = {}
        #re.sub(' +',' ',a)
        vals = Lines[1][line].split(',')
        for val in vals:
            dict_synonym_terms.update({re.sub(' +',' ',Lines[0][line].lower().strip()) : re.sub(' +',' ', val.lower().strip()) })
            dict_synonym_values.update({re.sub(' +',' ', val.lower().strip()) : re.sub(' +',' ',Lines[0][line].lower().strip()) })
            #print(len(str(Lines[5][line])))
            if str(Lines[5][line]) != 'nan':
                dict_synonym_values_adds.update({re.sub(' +',' ', val.lower().strip()) : re.sub(' +',' ',str(Lines[5][line]))})

  # resort dicts DESC
#    for k in sorted(dict_synonym_values, key=len, reverse=True):
#        dict_synonym_values_sorted[k] = dict_synonym_values[k]
#    dict_synonym_values_sorted = sorted(dict_synonym_values, key=lambda l: len(l[0]), reverse=True)
#    k = list(dict_synonym_values.items())
#    k.sort(key=lambda x:len(x[0]),reverse=True)
#    print('K:', k)

#    for i in k :
#        dict_synonym_values_sorted.update({i[0]:i[1]})
    return

class Suggestion:
    def __init__(self, art, name, dist, dist_updated):
        self.art = art
        self.name = name
        self.dist = dist
        self.dist_updated = dist_updated

@app.route("/")
def index():
   getDBLinksList()
   filename = ''
   return render_template('index.html', page=1, web_data = urls, html_data = '', web_data_len = len(urls),  temp='', file = '')   

@app.route("/chat")
def chat():
   return render_template('chat.html', page=1, web_data = True, html_data = '', web_data_len = len(urls),  temp='', file = '')   

@app.route("/chat2")
def chat2():
   return render_template('chat2.html', page=1, web_data = True, html_data = '', web_data_len = len(urls),  temp='', file = '')   

@app.route("/upload")
def  upload():
    global urls
    return render_template('upload.html', page=1, web_data = True, html_data = '', web_data_len = len(urls),  temp='', file = '')   

@app.route("/synonym")
def  synonum():
    global urls
    return render_template('synonym.html', page=1, web_data = True, html_data = '', web_data_len = len(urls),  temp='', file = '')   

@app.route("/slang")
def  slang_html():
    global urls
    return render_template('slang.html', page=1, web_data = True, html_data = '', web_data_len = len(urls),  temp='', file = '')   

@app.route("/slang", methods = ['POST'])
def  slang():
    global urls
    return uploadfile(False)
    #return render_template('slang.html', page=1, web_data = True, html_data = '', web_data_len = len(urls),  temp='', file = '')  

@app.route("/dicts/update")
def dicts_update():
    d = []
    d.append({"Object_id": 27168, "name": 'Zusatz_Rename_1'})
    d.append({"Object_id": 27144, "name": 'Zusatz_Rename_31'})
    d.append({"Object_id": 27145, "name": 'Zusatz_Rename_13'})
    d.append({"Object_id": 27146, "name": 'Zusatz_Rename_12'})
    d.append({"Object_id": 27147, "name": 'Zusatz_Rename_144'})
    d.append({"Object_id": 27148, "name": 'Zusatz_Rename_1332'})
    d.append({"Object_id": 27168, "name": 'Zusatz_Rename_123'})
    d.append({"Object_id": 27171, "name": 'Zusatz_Rename_1234'})
    return jsonify(d)

@app.route("/dicts/order", methods = ['GET'])
def dicts_order():
    return jsonify(request.args)

@app.route("/dicts/report", methods = ['GET'])
def dicts_report():
    return jsonify(request.args)

@app.route("/dicts")
def dicts():
    print(request.args)
    kind = request.args.get('kind', default = 0, type = int)
    list = request.args.get('list', default = 'null', type = str)

    d = []

    if kind == 0:
        d.append({"id": 0, "name": ''})
        d.append({"id": 100, "name": 'Tree'})
        d.append({"id": 101, "name": 'Bench'})
        d.append({"id": 103, "name": 'Container'})
        d.append({"id": 104, "name": 'Mobile'})

    if kind == 1 and list == "all":
        d.append({"id": 0, "name": ''})
        d.append({"id": 100, "name": 'Id'})
        d.append({"id": 101, "name": 'Name'})
        d.append({"id": 102, "name": 'Kind'})
        d.append({"id": 103, "name": 'Object_id'})
        d.append({"id": 104, "name": 'Author'})
        d.append({"id": 105, "name": 'Lat'})
        d.append({"id": 106, "name": 'Lng'})


        #d = "[{'id': 100, 'name': 'Tree'}, {'id': 101, 'name': 'Bench'}, {'id': 103, 'name': 'Container'}, {'id': 104, 'name': 'Mobile'}]"

    return jsonify(d)

def parse_lines(Lines, Type, isOrder = True, withSpec = True):
    dict = []
    i = 0
    count = 0
    print('isOrder2', isOrder)

    for line in Lines:
        #print("Line{}: {}".format(count, line.strip()))
        if line[:2] == '##':
            continue

        line = line.replace('\n', '')
        print(line)
        ret = []
        ret_spec = []
        ret_qty = []
        ret_woqty = []
        ret_syn = []
        ret_all = []

        ret_spec_arr = []
        ret_qty_arr = []
        ret_woqty_arr = []
        ret_syn_arr = []

        art_found = find_art(df, line.replace('\n', ''))
        answ_cnt = 0
        sug_ent_map = {}
        #print(len(art_found), art_found)
        dist_sum = 0 
        dist_res = [[], [], []]

        if len(art_found) > 0 and isOrder :
            print('art_found.art', art_found.values[0][0], art_found.values[0][1])
            ret.append({"art": art_found.values[0][0], "name": art_found.values[0][1], "dist": 0.001, "dist_updated": 0.001, "dist_updated2": 0.001, "qty": art_found.values[0][2], "spec": art_found.values[0][3], "found": 'by_art'})

        if len(line) > 0:
            count += 1
            try:
                orig_lang = detect(line)
            except:
                orig_lang = 'uk'

            print(orig_lang)
            if (orig_lang != 'uk'):
                line_ua = translate_deepl(line) #translate_text(line, "Ukrainian") #translate_google_trans(line) #translate_text(line, "Ukrainian") #translate_text_by_deepl(line) #translate_text_by_word(line) #translate_text(line, "Ukrainian")
                line_ua = line_ua.replace('Перекласти на українську: ', '')  
            else:
                line_ua = line

            chat_answ  = ""
            #print('PRINT:', line_ua)
            syns = replace_with_synonym(line_ua, False)

            orig_line_ua = line_ua
            line_ua = syns[0]
            #print('PRINT2:', new_line_ua, syns)
            syn_qtys = syns[1]
            
            if isOrder:
                response_data = answer_question(df_spec, question=line_ua, debug=False, max_len=1200, max_tokens=400, with_leaves = True)
                for key, r in enumerate(response_data):
                    answ_cnt +=1
                    r = response_data[key]
                    #if answ_cnt > 5: 
                    #    break
                    if True:#not r[3] in sug_ent_map:
                        #sug_ent_map.update({r[3]: r[6]})    

                        dist = recalc_dist(line_ua, r[0])
                        dist_res = recalc_dist2(line_ua, r[0])
                        dist2 = dist_res[0]
                        #print(locale.atof(str(r[6]).replace(",", ".")))
                        if locale.atof(str(r[6]).replace(",", ".")) > 0 and withSpec:
                            if len(dist_res[1]): # only if other word hear 
                                dist2 = dist2 * SPEC_KOEF
                                dist_res[1].append({"SPEC": SPEC_KOEF})
#                        if locale.atof(str(r[5]).replace(",", ".")) > 0:
#                            if len(dist_res[1]): # only if other word hear 
#                                dist2 = dist2 * LEAVES_KOEF
#                                dist_res[1].append({"LEAVES": LEAVES_KOEF})            
                        #sug = Suggestion(r[3], r[0], r[4], r[4] - dist)
                        ret_spec.append({"art": r[3], "name": r[0], "dist": r[4], "dist_updated": r[4] * dist2, "dist_hist": dist_res[1], "qty": r[5], "spec": r[6], "found": 'by_spec'}) #"dist_updated": r[4] - dist, 
                
                ret_spec = sorted(ret_spec, key=lambda d: d['dist_updated'])
                cnt = 0
                for key, r in enumerate(ret_spec):
                    if not r.get("art") in sug_ent_map:
                        sug_ent_map.update({r.get("art"): r.get("spec")})  
                        cnt+=1 
                        ret.append(r)
                        ret_spec_arr.append(r)
                        ret_all.append(r)
                    if cnt > 3:
                        break                   

            if isOrder:
                response_data = answer_question(df_leaves, question=line_ua, debug=False, max_len=1200, max_tokens=400, with_leaves = True)
                for key, r in enumerate(response_data):
                    answ_cnt +=1
                    r = response_data[key]
                    #if answ_cnt > 19: 
                    #    break
                    if True: #not r[3] in sug_ent_map:
                        #sug_ent_map.update({r[3]: r[6]})    

                        dist = recalc_dist(line_ua, r[0])
                        dist_res = recalc_dist2(line_ua, r[0])
                        dist2 = dist_res[0]
                        #print(locale.atof(str(r[6]).replace(",", ".")))
                        if locale.atof(str(r[6]).replace(",", ".")) > 0 and withSpec:
                            dist2 = dist2 * SPEC_KOEF
                            dist_res[1].append({"SPEC": SPEC_KOEF})
                        if locale.atof(str(r[5]).replace(",", ".")) > 0:
                            dist2 = dist2 * LEAVES_KOEF
                            dist_res[1].append({"LEAVES": LEAVES_KOEF})      
                        #sug = Suggestion(r[3], r[0], r[4], r[4] - dist)
                        ret_qty.append({"art": r[3], "name": r[0], "dist": r[4], "dist_updated": r[4] * dist2, "dist_hist": dist_res[1], "qty": r[5], "spec": r[6], "found": 'by_leaves'}) #  "dist_updated": r[4] - dist, 
                
                ret_qty = sorted(ret_qty, key=lambda d: d['dist_updated'])
                cnt = 0
                for key, r in enumerate(ret_qty):
                    if not r.get("art") in sug_ent_map:
                        sug_ent_map.update({r.get("art"): r.get("spec")})  
                        cnt+=1 
                        ret.append(r)
                        ret_qty_arr.append(r)
                        ret_all.append(r)                        
                    if cnt > 3:
                        break               

            if isOrder:
                response_data = answer_question(df, question=line_ua, debug=False, max_len=1200, max_tokens=400)
                dist_sum = 0
                for key, r in enumerate(response_data):
                    answ_cnt +=1
                    r = response_data[key]
                    # if answ_cnt > 26: 
                    #     break

                    if True: #not r[3] in sug_ent_map:
                        #sug_ent_map.update({r[3]: r[6]})   

                        dist = recalc_dist(line_ua, r[0])
                        dist_res = recalc_dist2(line_ua, r[0], True)
                        dist2 = dist_res[0]
                        if locale.atof(str(r[6]).replace(",", ".")) > 0 and withSpec:
                            dist2 = dist2 * SPEC_KOEF
                            dist_res[1].append({"SPEC": SPEC_KOEF})
#                        if locale.atof(str(r[5]).replace(",", ".")) > 0:
#                            dist2 = dist2 * LEAVES_KOEF
#                            dist_res[1].append({"LEAVES": LEAVES_KOEF})  
                        #print(locale.atof(str(r[6]).replace(",", ".")))
                        #sug = Suggestion(r[3], r[0], r[4], r[4] - dist)
                        ret_woqty.append({"art": r[3], "name": r[0], "dist": r[4], "dist_updated": r[4] * dist2, "dist_hist": dist_res[1], "qty": r[5], "spec": r[6], "found": 'by_all'}) # "dist_updated": r[4] - dist, 
                    #ret.append(sug)

                ret_woqty = sorted(ret_woqty, key=lambda d: d['dist_updated']) 
                #print(ret_woqty)
                cnt = 0
                for key, r in enumerate(ret_woqty):
                    if not r.get("art") in sug_ent_map:
                        sug_ent_map.update({r.get("art"): r.get("spec")})  
                        cnt+=1 
                        dist_sum += r.get("dist")
                        ret.append(r)
                        ret_woqty_arr.append(r)
                        ret_all.append(r)                        
                    if cnt > 4:
                        break

                #print(r)


            if  ((dist_sum/5 > 0.5 and not len(art_found) > 0 and isOrder) or (line_ua != orig_line_ua and isOrder)) and False: #len(dist_res[2]):
                print(dist_sum, dist_sum/5, line_ua, dist_res[2])
                #syns = replace_with_synonym(line_ua)

                #new_line_ua = syns[0]
                #syn_qtys = syns[1]

                #print('synonym:', line_ua, new_line_ua, syn_qtys)

                if new_line_ua == line_ua:                
                    chat_answ = new_line_ua#ask_chat(question=line_ua)
                else:
                    chat_answ = new_line_ua    

                #print('chat_answ', chat_answ)
                #chat_answ = "етикетка на аркуші"
                response_data = answer_question(df, question=chat_answ, debug=False, max_len=1200, max_tokens=400)
                #print(response_data)

                for key, r in enumerate(response_data):
                    answ_cnt +=1
                    r = response_data[key]
                    # if answ_cnt > 32: 
                    #     break
                    if True:#not r[3] in sug_ent_map:
                    #    sug_ent_map.update({r[3]: r[6]})   

                        dist = recalc_dist(chat_answ, r[0])
                        dist_res = recalc_dist2(chat_answ, r[0], True, syn_qtys)
                        dist2 = dist_res[0]
                        if locale.atof(str(r[6]).replace(",", ".")) > 0:
                            dist2 = dist2 * SPEC_KOEF
                            dist_res[1].append({"SPEC": 0.5})
                        ret_syn.append({"art": r[3], "name": r[0], "dist": r[4], "dist_updated": r[4] - dist, "dist_updated2": r[4] * dist2, "dist_hist": dist_res[1], "qty": r[5], "spec": r[6], "found": 'by_syn'})


                ret_syn = sorted(ret_syn, key=lambda d: d['dist_updated2']) 
                #print(ret_woqty)
                cnt = 0
                for key, r in enumerate(ret_syn):
                    if not r.get("art") in sug_ent_map:
                        sug_ent_map.update({r.get("art"): r.get("spec")})  
                        cnt+=1 
                        ret.append(r)
                        ret_syn_arr.append(r)                        
                    if cnt > 4:
                        break

                    #print(r)

                val = {"row" : count, "orig": line, "orig_ua": line_ua, "name_syn": chat_answ, "lng": orig_lang.upper(),   "suggestions": { "spec": ret_spec_arr, "leaves" : ret_qty_arr, "all" : ret_woqty_arr, "syn": ret_syn_arr } }
                dict.append(val)                
                print("Синоним для ", line_ua, ": ", chat_answ)
            else:
                if not isOrder:
                    chat_answ = line_ua    

                    #print('chat_answ', chat_answ)
                    #chat_answ = "етикетка на аркуші"
                    new_line_ua = parse_units(line_ua.lower().replace(',', ' ').replace('(', ' ').replace(')', ' ').replace('"', ' ').strip())
                    for word in new_line_ua.strip().split():
                        word_met = 0
                        if not word in prepos_dict.keys() and not word.isnumeric() and len(word) > 1:
                            if not df['text'].str.contains(word, regex=False).any():
                                response_data = answer_question(df, question=word, debug=False, max_len=1200, max_tokens=400)
                                for key, r in enumerate(response_data):
                                    answ_cnt +=1
                                    r = response_data[key]
                                    answ_line_normal = ''
                                    #normalizing
                                    #word = word.replace(',', ' ').replace('(', ' ').replace(')', ' ').strip()
                                    answ_line = r[0]
                                    answ_line = parse_units(answ_line.lower().replace(',', ' ').replace('(', ' ').replace(')', ' ').replace('"', ' ').strip())
                                    
                                    #  print('parse', order_line, answer_line)
                                    for key, word2 in enumerate(answ_line.split()):
                                        answ_line_normal+= morph.parse(word2)[0].normal_form + ' '

                                    #print(word, r[0])
                                    word = morph.parse(word)[0].normal_form
                                    if word in answ_line.strip().lower():
                                        word_met +=1

                                if word_met != 0:
                                    pass
                                    #ret_syn.append({"word": word, "dist_updated2": 1 / word_met, "count": word_met})
                                else:
                                    ret_syn.append({"word": word, "dist_updated": 1000, "count": 0, "isSLANG": True})  
                            else:
                                 #ret_syn.append({"word": word, "dist_updated2": 0.001, "count": 1, "isSLANG": 'Found'})  
                                 pass          
                        else:
                                pass
                                #ret_syn.append({"word": word, "dist_updated2": 1 / 100 - key, "count": 100, "is_prep": True,  "found": 'slangs'})
                        #print('WORD', word)
 #                       print(ret_syn)

                    ret_syn = sorted(ret_syn, key=lambda d: d['dist_updated']) 
                    #print(ret_woqty)
                    cnt = 0
                    for key, r in enumerate(ret_syn):
                        ret_syn_arr.append(r)                        
                            #if cnt > 4:
                            #    break

                        #print(r)

                    val = {"row" : count, "orig": line, "orig_ua": line_ua, "name_syn": chat_answ, "lng": orig_lang.upper(),   "suggestions": { "spec": ret_spec_arr, "leaves" : ret_qty_arr, "all" : ret_woqty_arr, "syn": ret_syn_arr } }
                    dict.append(val)                
                    print("сленги для ", line_ua, ": ", chat_answ)                    
                else:
                    ret_all = sorted(ret_all, key=lambda d: d['dist_updated']) 
                    val = {"row" : count, "orig": line, "orig_ua": line_ua, "lng": orig_lang.upper(),   "suggestions": ret_all} #"suggestions": { "spec": ret_spec_arr, "leaves" : ret_qty_arr, "all" : ret_woqty_arr, "syn": ret_syn_arr }}
                    dict.append(val)
                    print(line_ua, dist_sum/5)



    return dict

@app.route("/synonym", methods=['POST'])
def  uploadsynonymfile():
    global urls
    #list = []
    dict = []
    i = 0
    ts = str(time.time())
    
    if request.method == 'POST':   
        f = request.files['file'] 
        f.save('./processed/synonyms/' + f.filename)  
        count = 0
        is_file = False
        
        # xls          
        if (f.filename.split('.')[-1]=='xls' or f.filename.split('.')[-1]=='xlsx'):
            is_file = True
            df_file = pd.read_excel('./processed/synonyms/' + f.filename, header=None, skiprows=[0])
            #print(df_file, df_file[0].tolist())
            dict = parse_lines_synonym(df_file) 
            #ts = str(ts) + "_xls"

    # #print(dict)
    # with open('./uploaded/' + ts+".json", "w", encoding="utf-8") as outfile: 
    #     json.dump(dict, outfile, ensure_ascii=False)

    # if is_file:
    #     #flash('File prepared')
    #     return send_file('./uploaded/' + ts+".json", as_attachment=True)  
    # else: 
    return render_template('synonym.html', page=1, web_data = True, html_data = '', web_data_len = len(urls),  temp='', file = f)   

@app.route("/upload", methods=['POST'])
def  uploadfile(isOrder = True):
    global urls
    #list = []
    dict = []
    i = 0
    ts = str(time.time())
    
    print('isOrder', isOrder, request.form['action'])
    withSpec = True
    if request.form['action'] == 'UploadWOSpec':
        withSpec = False

    if request.method == 'POST':   
        f = request.files['file'] 
        f.save('./uploaded/' + f.filename)  
        count = 0
        is_file = False
        # txt
        if (f.filename.split('.')[-1]=='txt'):
            is_file = True
            file1 = open('./uploaded/' + f.filename,  mode="r", encoding="utf-8")
            Lines = file1.readlines()
#            print(Lines)
#            Lines = translate_text(Lines, "Ukrainian")
#            print(Lines)  

            dict = parse_lines(Lines, 'txt', isOrder, withSpec)            
            ts = str(ts) + "_txt"

        # doc        
        if (f.filename.split('.')[-1]=='doc' or f.filename.split('.')[-1]=='docx'):
            is_file = True
            text = textract.process('./uploaded/' + f.filename)
            #Lines = text.readlines()
            text = text.decode("utf-8")
            #text = text.split('\n')
 
            dict = parse_lines(text.split('\n'), 'doc', isOrder, withSpec) 
            ts = str(ts) + "_doc"

        # xls          
        if (f.filename.split('.')[-1]=='xls' or f.filename.split('.')[-1]=='xlsx'):
            is_file = True
            df_file = pd.read_excel('./uploaded/' + f.filename, header=None)

            dict = parse_lines(df_file[0].tolist(), 'xls', isOrder, withSpec) 
            ts = str(ts) + "_xls"

    #print(dict)

    if isOrder: 
        with open('./uploaded/' + ts+".json", "w", encoding="utf-8") as outfile:
            json.dump(dict, outfile, ensure_ascii=False)
        outfile.close()    
    else:
        #print('dict:', dict)
        write_rows = []
        write_rows.append(["word", "orig", "row"])
        for obj in dict:
            #print('obj0', obj)
            obj = eval(str(obj))#json.loads(str(obj)) # 
            if len(obj.get("suggestions").get('syn')): 
                for word in obj.get("suggestions").get("syn"):
                    print ("word:", word, word.get("word"), obj.get("orig"), str(obj.get("row")))
                    write_rows.append([word.get("word"), obj.get("orig"), obj.get("row")])
                    #writer.writerow(word.get("word") + ',' + obj.get("orig") + ',' + str(obj.get("row")))
        #print('write_rows', write_rows)            
        with open('./uploaded/' + ts+'.csv', "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(write_rows) 

    #print('is_file', is_file, ts)

    if is_file:
        #flash('File prepared')
        print('Uploaded', './uploaded/' + ts+".json")
        
        #return send_file('./uploaded/1711314090.482361_txt.json', as_attachment=True)
        try: 
            if isOrder:
                return send_file('./uploaded/' + ts+".json", as_attachment=True)  
            else:
                return send_file('./uploaded/' + ts+".csv", as_attachment=True) 
        except Error:
            print(Error)

        if isOrder:
            return render_template('upload.html', page=1, web_data = True, html_data = '', web_data_len = len(urls),  temp='', file = f)  
        else:
            return render_template('slang.html', page=1, web_data = True, html_data = '', web_data_len = len(urls),  temp='', file = f)
    else: 
        return render_template('upload.html', page=1, web_data = True, html_data = '', web_data_len = len(urls),  temp='', file = f)   

@app.route("/crawl")
def crawling():
    global urls
    urls = crawl(full_url)
    return render_template('index.html', page=1, web_data = urls, html_data = '', web_data_len = len(urls),  temp='', file = '')   

@app.route("/crawl_ku")
def crawling_ku():
    global urls

    urls = crawl_ku(full_url_ku)
    #return "<p>Got it!</p>"
    return render_template('index.html', page=1, web_data = urls, html_data = '', web_data_len = len(urls),  temp='', file = '')  

@app.route("/maintenance")
def  maintenance():
    global urls
    urls = crawl(full_url)
    token()
    embedding()
    return render_template('index.html', page=1, web_data = urls, html_data = '', web_data_len = len(urls),  temp='', file = '')   

@app.route("/post", methods=['POST'])
def post():
    global df
    withSpec = True
    #request_data = request.get_json();
    request_data = json.loads(request.json)['msg']
    if 'embeddings' not in df:
        df=pd.read_csv('processed/embeddings_ku.csv', index_col=0)
        df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

    # Что можешь сказать о распределении работ?
    print('response')
    request_data = request_data.lower().strip()
    response_data = answer_question(df, question=request_data, debug=False, max_len=1200, max_tokens=400)
    ret = []
    for r in response_data:
        ret.append({"a": r[0], "u": r[1], "t": r[2]})
    #print(response_data)
    #return jsonify({"a": response_data[0], "u": response_data[1]})
    return jsonify(ret)

@app.route("/api/order", methods=['POST'])
def api_order(key=None):
    global df
    withSpec = True
    isOrder = True
    #request_data = request.get_json();
    #order  = request.args.get('order', None)
    #syns  = request.args.get('syns', None)
    #json.dumps(a).encode("latin-1")
    #request.data = json.dumps(request.data).encode("utf-8")
    #print(str(json.loads(request.data.decode('utf-8'))))
    #return 'success', 200
    headers = get_headers_as_dict(str(request.headers))

    request_data = "" #json.loads(request.data)['msg']
    if 'Authorization' in headers:

        if request.headers.get('Authorization').split()[1] != api_token:
            return jsonify({"status": "error", "route": "order", "error_desc": "wrong token"}), 200
        #else:
        #    return jsonify({"status": "success", "route": "order", "error_desc": "", "msg": request_data}), 200
    else:
        return jsonify({"status": "error", "route": "order", "error_desc": "wrong token"}), 200


    print('request_data', request_data)
    #if 'embeddings' not in df:
    #    df=pd.read_csv('processed/embeddings_ku.csv', index_col=0)
    #    df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

    #print('response')
    #request_data = request_data.lower().strip()
    #response_data = answer_question(df, question=request_data, debug=False, max_len=1200, max_tokens=400)
    #ret = []
    #for r in response_data:
    #    ret.append({"a": r[0], "u": r[1], "t": r[2]})
    #print(response_data)
    #return jsonify({"a": response_data[0], "u": response_data[1]})

    global urls
    #list = []
    dict = []
    i = 0
    ts = str(time.time())
    

    if request.method == 'POST':   
        f = request.files['file'] 
        f.save('./uploaded/' + f.filename) 
        print('FILE', f.filename) 
        count = 0
        is_file = False
        # txt
        if (f.filename.split('.')[-1]=='txt'):
            is_file = True
            file1 = open('./uploaded/' + f.filename,  mode="r", encoding="utf-8")
            Lines = file1.readlines()
#            print(Lines)
#            Lines = translate_text(Lines, "Ukrainian")
#            print(Lines)  

            dict = parse_lines(Lines, 'txt', isOrder, withSpec)            
            ts = str(ts) + "_txt"

        # doc        
        if (f.filename.split('.')[-1]=='doc' or f.filename.split('.')[-1]=='docx'):
            is_file = True
            text = textract.process('./uploaded/' + f.filename)
            #Lines = text.readlines()
            text = text.decode("utf-8")
            #text = text.split('\n')
 
            dict = parse_lines(text.split('\n'), 'doc', isOrder, withSpec) 
            ts = str(ts) + "_doc"

        # xls          
        if (f.filename.split('.')[-1]=='xls' or f.filename.split('.')[-1]=='xlsx'):
            is_file = True
            df_file = pd.read_excel('./uploaded/' + f.filename, header=None)

            dict = parse_lines(df_file[0].tolist(), 'xls', isOrder, withSpec) 
            ts = str(ts) + "_xls"

    #print(dict)

    return jsonify({"status": "success", "route": "order", "error_desc": "", "msg": request_data, "data": json.dumps(dict, ensure_ascii=False)}), 200 

    if isOrder: 
        with open('./uploaded/' + ts+".json", "w", encoding="utf-8") as outfile:
            json.dump(dict, outfile, ensure_ascii=False)
        outfile.close()    
    else:
        write_rows = []
        write_rows.append(["word", "orig", "row"])
        for obj in dict:
            #print('obj0', obj)
            obj = eval(str(obj))#json.loads(str(obj)) # 
            if len(obj.get("suggestions").get('syn')): 
                for word in obj.get("suggestions").get("syn"):
                    print ("word:", word, word.get("word"), obj.get("orig"), str(obj.get("row")))
                    write_rows.append([word.get("word"), obj.get("orig"), obj.get("row")])
                    #writer.writerow(word.get("word") + ',' + obj.get("orig") + ',' + str(obj.get("row")))
        #print('write_rows', write_rows)            
        with open('./uploaded/' + ts+'.csv', "w", newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(write_rows) 

    #print('is_file', is_file, ts)

    if is_file:
        #flash('File prepared')
        print('Uploaded', './uploaded/' + ts+".json")
        
        #return send_file('./uploaded/1711314090.482361_txt.json', as_attachment=True)
        try: 
            if isOrder:
                return send_file('./uploaded/' + ts+".json", as_attachment=True)  
            else:
                return send_file('./uploaded/' + ts+".csv", as_attachment=True) 
        except Error:
            print(Error)

        if isOrder:
            return render_template('upload.html', page=1, web_data = True, html_data = '', web_data_len = len(urls),  temp='', file = f)  
        else:
            return render_template('slang.html', page=1, web_data = True, html_data = '', web_data_len = len(urls),  temp='', file = f)
    else: 
        return render_template('upload.html', page=1, web_data = True, html_data = '', web_data_len = len(urls),  temp='', file = f)   










@app.route("/post2", methods=['POST'])
def post2():
    global df
    #request_data = request.get_json();
    request_data = json.loads(request.json)['msg']

    # Что можешь сказать о распределении работ?
    print('response2')
    request_data = request_data.lower().strip()
    response_data = answer_question(df, question=request_data, debug=False, max_len=1200, max_tokens=400, isNewEmbed=True)
    ret = []
    for r in response_data:
        ret.append({"a": r[0], "u": r[1], "t": r[2]})
    #print(response_data)
    #return jsonify({"a": response_data[0], "u": response_data[1]})
    return jsonify(ret)

def getDBLinksList():
    global urlsindb, urls
    connection = mysql.connector.connect( user='u_reslyete', password = 'htmTmsLT' , host = "127.0.0.1" , database = 'reslyete')

    cursor = connection.cursor()
    #cursor.execute("select id, url from links")
    cursor.execute("select id, guid, name from	goods where name like '%миття%'")
    # get all records and save to Set
    getLinks = cursor.fetchall()
    for row in getLinks:
        urlsindb.add(row[2])
    urls = urlsindb
    cursor.close()
    connection.close()

def get_headers_as_dict(headers: str) -> dict:
    dic = {}
    for line in headers.split("\n"):
        if line.startswith(("GET", "POST")):
            continue
        point_index = line.find(":")
        dic[line[:point_index].strip()] = line[point_index+1:].strip()
    return dic


# Regex pattern to match a URL
# Create a class to parse the HTML and get the hyperlinks
class HyperlinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        # Create a list to store the hyperlinks
        self.hyperlinks = []

    # Override the HTMLParser's handle_starttag method to get the hyperlinks
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        # If the tag is an anchor tag and it has an href attribute, add the href attribute to the list of hyperlinks
        if tag == "a" and "href" in attrs:
            self.hyperlinks.append(attrs["href"])

# Function to get the hyperlinks from a URL
def get_hyperlinks(url):

    # Try to open the URL and read the HTML
    try:
        # Open the URL and read the HTML
        with urllib.request.urlopen(url) as response:

            # If the response is not HTML, return an empty list
            if not response.info().get('Content-Type').startswith("text/html"):
                return []

            # Decode the HTML
            html = response.read().decode('utf-8')
    except Exception as e:
        print(e)
        return []

    # Create the HTML Parser and then Parse the HTML to get hyperlinks
    parser = HyperlinkParser()
    parser.feed(html)

    return parser.hyperlinks

# Function to get the hyperlinks from a URL that are within the same domain
def get_domain_hyperlinks(local_domain, url):
    clean_links = []
    for link in set(get_hyperlinks(url)):
        clean_link = None

        # If the link is a URL, check if it is within the same domain
        if re.search(HTTP_URL_PATTERN, link):
            # Parse the URL and check if the domain is the same
            url_obj = urlparse(link)
            if url_obj.netloc == local_domain:
                clean_link = link

        # If the link is not a URL, check if it is a relative link
        else:
            if link.startswith("/"):
                link = link[1:]
            elif link.startswith("#") or link.startswith("mailto:"):
                continue
            clean_link = "https://" + local_domain + "/" + link

        if clean_link is not None:
            if clean_link.endswith("/"):
                clean_link = clean_link[:-1]
#            if clean_link.find('User_Guide') > 0:
            clean_links.append(clean_link)

    # Return the list of hyperlinks that are within the same domain
    return list(set(clean_links))

def crawl(url):
    global connection
    global cursor
    global ulrsinbd 
    # Parse the URL and get the domain
    local_domain = urlparse(url).netloc
     # Create a queue to store the URLs to crawl
    queue = deque([url])

    # Create a set to store the URLs that have already been seen (no duplicates)
    seen = set([url])

    # Create a directory to store the text files
    if not os.path.exists("text/"):
            os.mkdir("text/")

    if not os.path.exists("text/"+local_domain+"/"):
            os.mkdir("text/" + local_domain + "/")

    # Create a directory to store the csv files
    if not os.path.exists("processed"):
            os.mkdir("processed")

    i = 0

 #   if len(urlsindb) > 220:
 #       return urlsindb    

   # mySql_insert_query = """INSERT INTO Laptop (Id, Name, Price, Purchase_date) 
   #                         VALUES (%s, %s, %s, %s) """

   # record = (id, name, price, purchase_date)
   # cursor.execute(mySql_insert_query, record)
   # connection.commit()

    # While the queue is not empty, continue crawling
    while queue : #  and i < 300
        i+=1
        # Get the next URL from the queue
        url = queue.pop()
        print(url) # for debugging and to see the progress

        # Save text from the url to a <url>.txt file
        with open('text/'+local_domain+'/'+url[8:].replace("/", "_") + ".txt", "w", encoding="utf-8") as f:

            # Get the text from the URL using BeautifulSoup
            req = requests.get(url)
            soup = BeautifulSoup(req.content, "html.parser")
            text = ''

            if req.status_code != 404:
              if not soup.body.template.div is None :
                  text = BeautifulSoup(str(soup.body.template.div), "html.parser").text

            # If the crawler gets to a page that requires JavaScript, it will stop the crawl
            if ("You need to enable JavaScript to run this app." in text):
                print("Unable to parse page " + url + " due to JavaScript being required")

            # Otherwise, write the text to the file in the text directory
            f.write(text)

        # Get the hyperlinks from the URL and add them to the queue
        for link in get_domain_hyperlinks(local_domain, url):
            if link not in seen:
                queue.append(link)
                seen.add(link)
                if link not in urlsindb:
                    #mySql_insert_query = f'INSERT INTO Links (url) VALUES ("{link}")'
                    cursor.execute(f'INSERT INTO Links (url) VALUES ("{link}")')
                    connection.commit()
    return seen

def crawl_ku(url):
    global connection
    global cursor
    global ulrsinbd 
    # Parse the URL and get the domain
    local_domain = urlparse(url).netloc
     # Create a queue to store the URLs to crawl
    queue = deque([url])

    # Create a set to store the URLs that have already been seen (no duplicates)
    seen = set([url])

    # Create a directory to store the text files
    if not os.path.exists("text/"):
            os.mkdir("text/")

    if not os.path.exists("text/"+local_domain+"/"):
            os.mkdir("text/" + local_domain + "/")

    # Create a directory to store the csv files
    if not os.path.exists("processed"):
            os.mkdir("processed")

    i = 0

 #   if len(urlsindb) > 220:
 #       return urlsindb    

 #   mySql_insert_query = """INSERT INTO Laptop (Id, Name, Price, Purchase_date) 
 #                          VALUES (%s, %s, %s, %s) """
    connection = mysql.connector.connect( user='u_reslyete', password = 'htmTmsLT' , host = "127.0.0.1" , database = 'reslyete')
    cursor = connection.cursor()

    sql = forge_sql # """ select id, guid, name from	goods where name like '%миття%' """

   # record = (id, name, price, purchase_date)
    cursor.execute(sql)
    rs = cursor.fetchall()
    df_e = pd.read_excel(io=r"C:\Users\Paul\Downloads\ku_goods.xlsx")
    """

        #print(df_e.head(5))  # print first 5 rows of the dataframe

        for i2 in range(len(df_e)):

            #print(df_e.iloc[i2, 0], df_e.iloc[i2, 2])

            if (i > 0):
                with open('text/'+local_domain+'/g'+ str(df_e.iloc[i2, 2]) + ".txt", "w", encoding="utf-8") as f: #  + '_' + row[2].replace("/", "_").replace(".", "_").replace(",", "_") +

                    # Get the text from the URL using BeautifulSoup
                    text = ''

                    text = df_e.iloc[i2, 0]

                    # If the crawler gets to a page that requires JavaScript, it will stop the crawl
                    if ("You need to enable JavaScript to run this app." in text):
                        print("Unable to parse page " + url + " due to JavaScript being required")
                    
                    #print(text)

                    if df_e.iloc[i2, 0] not in seen:
                        queue.append(df_e.iloc[i2, 0])
                        seen.add(df_e.iloc[i2, 0])

                    # Otherwise, write the text to the file in the text directory
                    f.write(text)

            i+=1
    """
    """
        for filename in os.listdir('text/'+local_domain):
                if (i > 0):
                    with open('text/'+local_domain+'/'+ filename, "w", encoding="utf-8") as f: 

                        # Get the text from the URL using BeautifulSoup
                        text = ''

                        text = f.readline().lower()

                        # If the crawler gets to a page that requires JavaScript, it will stop the crawl
                        if ("You need to enable JavaScript to run this app." in text):
                            print("Unable to parse page " + url + " due to JavaScript being required")
                        
                        #print(text)

                        if df_e.iloc[i2, 0] not in seen:
                            queue.append(df_e.iloc[i2, 0])
                            seen.add(df_e.iloc[i2, 0])

                        # Otherwise, write the text to the file in the text directory
                        f.write(text)

            i+=1
    """
    for index, row in df.iterrows():
    #    print(index, df.at[index, 'embeddings']) # df.iloc[[index]]
    #    arr = pd.DataFrame(row).T.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
    #    print('arr', arr.iloc[0])
        i+= 1
        with open('text/'+local_domain+'/'+ df.at[index, 'art'] + '.txt', "w", encoding="utf-8") as f: 
            text = df.at[index, 'text']   

            if df.at[index, 'art'] not in seen:
                queue.append(df.at[index, 'art'])
                seen.add(text) 

            f.write(text)        
        #print('test', df.at[index, 'embeddings'][0:9], len(df.at[index, 'embeddings']))

        #f = os.path.join(directory, filename)
        # checking if it is a file
        #if os.path.isfile(f):
        #    print(f)
    print('Crawled')        
    return seen

    with open(r"C:\Users\Paul\Documents\ku_goods.csv", newline='', encoding="cp1251", errors='ignore') as csvfile:
        fieldnames = ['Name', 'Parent', 'Article', 'Unit']
        rs = csv.reader(csvfile, delimiter=';', quotechar='|')
        for row in rs:

            if (i > 0):
                with open('text/'+local_domain+'/g'+ str(row[2]) + ".txt", "w", encoding="utf-8") as f: #  + '_' + row[2].replace("/", "_").replace(".", "_").replace(",", "_") +

                    # Get the text from the URL using BeautifulSoup
                    text = ''

                    text = row[0]

                    # If the crawler gets to a page that requires JavaScript, it will stop the crawl
                    if ("You need to enable JavaScript to run this app." in text):
                        print("Unable to parse page " + url + " due to JavaScript being required")
                    
                    #print(text)

                    if row[0] not in seen:
                        queue.append(row[0])
                        seen.add(row[0])

                    # Otherwise, write the text to the file in the text directory
                    f.write(text)
            i+=1
    return seen

    for row in rs:
        #print "%s, %s" % (row["name"], row["category"])
        i+=1
        # Get the next URL from the queue
        # url = queue.pop()
        #print(row[2])
        with open('text/'+local_domain+'/g'+ str(row[0]) + ".txt", "w", encoding="utf-8") as f: #  + '_' + row[2].replace("/", "_").replace(".", "_").replace(",", "_") +

            # Get the text from the URL using BeautifulSoup
            text = ''

            text = row[2]

            # If the crawler gets to a page that requires JavaScript, it will stop the crawl
            if ("You need to enable JavaScript to run this app." in text):
                print("Unable to parse page " + url + " due to JavaScript being required")
            
            #print(text)

            if row[0] not in seen:
                queue.append(row[0])
                seen.add(row[0])

            # Otherwise, write the text to the file in the text directory
            f.write(text)

    cursor.close()
    connection.close()
        
    return seen

def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie

max_tokens = 500

@app.route("/token")
def token():
    global df
    global urls
    global tokenizer
    # Create a list to store the text files
    texts=[]

    #print('token', os.listdir("text/" + domain + "/"))
    #exit();
     
    # Get all the text files in the text directory
    
    for file in os.listdir("text/" + domain + "/"):

        # Open the file and read the text
        with open("text/" + domain + "/" + file, "r", encoding="utf8") as f:
            text = f.read()

            # Omit the first 11 lines and the last 4 lines, then replace -, _, and #update with spaces.
    #        texts.append((file[11:-4].replace('-',' ').replace('_', ' ').replace('#update',''), text))
            texts.append((file[:-4].replace('-',' ').replace('_', ' ').replace('#update',''), text))

    
    # Create a dataframe from the list of texts
    df2 = pd.DataFrame(texts, columns = ['art', 'text'])
    df2["text_orig"] = df2['text']#.apply(lambda x: x.lower().replace('"', " ").strip())
    df2["text"] = df2['text'].apply(lambda x: x.lower().replace('"', " ").strip())
     #df.head()
    # Set the text column to be the raw text with the newlines removed
    #print(df.text + ' !!! ' + df.fname.str.replace('g', '')) #df.fname.replace('g', ''))
    #print(df.text.str.replace('"', ''))
    text = remove_newlines(df2.text.str.replace('"', '').str.replace(' д/', ' для '))
    # text = text.str.replace(' д ', ' ').str.replace(' для ', ' ')
    df2['text'] = text # remove_newlines(df.text.str.replace('"', '').str.replace('/', ' ')) # + ' [' + df.fname.replace('g', '') + "]."; #df.fname + ". " + remove_newlines(df.text)
    #df.to_csv('processed/scraped_ku.csv')
    
     #df.head()

    # Load the cl100k_base tokenizer which is designed to work with the ada-002 model
    # tokenizer = tiktoken.get_encoding("cl100k_base")
    
    #df = pd.read_csv('processed/scraped_ku.csv', index_col=0)
 
    #df.columns = ['title', 'text']
    #df.columns = ['art', 'text']
    
    # Tokenize the text and save the number of tokens to a new column
    df2['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    df2.to_csv('processed/scraped_ku.csv')

    print(df.head())

    shortened = []

    # Loop through the dataframe
    for row in df.iterrows():

        # If the text is None, go to the next row
        if row[1]['text'] is None:
            continue

        # If the number of tokens is greater than the max number of tokens, split the text into chunks
        if row[1]['n_tokens'] > max_tokens:
            shortened += split_into_many(row[1]['text'], max_tokens, row[1]['art'])

        # Otherwise, add the text to the list of shortened texts
        else:
            shortened.append( [ row[1]['text'], row[1]['art'] ])


    #df = pd.DataFrame(shortened, columns = ['text', 'art'])
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

    #print(df.head())
    print('Tokenized')
    #df.to_pickle(path='processed/embeddings_ku.pkl', compression='gzip')
    df = df.drop(columns=['qty_leaves', 'price_spec'])	
    df.to_csv('processed/embeddings_ku.csv')
    print('Saved tokenized')
    return render_template('index.html', page=1, web_data = urls, html_data = '', web_data_len = len(urls),  temp='', file = '')   

def token_checknull(df):
    #global df
    global urls
    global tokenizer
    # Create a list to store the text files

    # Load the cl100k_base tokenizer which is designed to work with the ada-002 model
    # tokenizer = tiktoken.get_encoding("cl100k_base")
    
    #df = pd.read_csv('processed/scraped_ku.csv', index_col=0)
 
    # Tokenize the text and save the number of tokens to a new column
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

    #print(df.head())

    shortened = []

    # Loop through the dataframe
    for row in df.iterrows():

        # If the text is None, go to the next row
        if row[1]['text'] is None:
            continue

        # If the number of tokens is greater than the max number of tokens, split the text into chunks
        if row[1]['n_tokens'] > max_tokens:
            shortened += split_into_many(row[1]['text'])

        # Otherwise, add the text to the list of shortened texts
        else:
            shortened.append( row[1]['text'] )


    df = pd.DataFrame(shortened, columns = ['text'])
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

    #print(df.head())
    #return render_template('index.html', page=1, web_data = urls, html_data = '', web_data_len = len(urls),  temp='', file = '')   

    # Visualize the distribution of the number of tokens per row using a histogram
    #df.n_tokens.hist()

# Function to split the text into chunks of a maximum number of tokens
def split_into_many(text, max_tokens = max_tokens, art = ''):
    #tokenizer = tiktoken.get_encoding("cl100k_base")
    global tokenizer
    # Split the text into sentences
    sentences = text.split('. ')

    # Get the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]

    chunks = []
    tokens_so_far = 0
    chunk = []

    # Loop through the sentences and tokens joined together in a tuple
    for sentence, token in zip(sentences, n_tokens):

        # If the number of tokens so far plus the number of tokens in the current sentence is greater
        # than the max number of tokens, then add the chunk to the list of chunks and reset
        # the chunk and tokens so far
        if tokens_so_far + token > max_tokens:
            chunks.append([". ".join(chunk) + "." , art])
            chunk = []
            tokens_so_far = 0

        # If the number of tokens in the current sentence is greater than the max number of
        # tokens, go to the next sentence
        if token > max_tokens:
            continue

        # Otherwise, add the sentence to the chunk and add the number of tokens to the total
        chunk.append(sentence)
        tokens_so_far += token + 1

    return chunks


#print("HERE!!!")
#exit()

#print(df.head())
#df.n_tokens.hist()

#print(df.head(10), len(df))
#dfs = np.array_split(df, 3)
#exit()
openai.api_key = apik 
#dfs[0]['embeddings'] = dfs[0].text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
#dfs[0].to_csv('processed/embeddings1.csv')
#dfs[0].head()
#print(df.text)

#!!!!
if 'embeddings' not in df:
    df['embeddings'] = ''
#!!!!p

@app.route("/embedding")
def embedding():
    global df
    global np
    global reInit
    #print(df.head())
    if 'embeddings' not in df:
        df['embeddings'] = ''


    i = 0
    #df = df[:20]
    #print(df)
    if reInit:
        df['embeddings'] = np.array2string((np.full((1536), -100.0)), separator=",")
    #print()
    for index, row in df.iterrows():
    #    print(index, df.at[index, 'embeddings']) # df.iloc[[index]]
    #    arr = pd.DataFrame(row).T.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
    #    print('arr', arr.iloc[0])
        i+= 1
 #       if (i > 20):
 #           break

#        print('test', df.at[index, 'embeddings'][0:9], len(df.at[index, 'embeddings']))
        #if (i < 10):
        if (True):
            if ((df.at[index, 'embeddings'][0:13] == '[-100.,-100.,')): # (len(df.at[index, 'embeddings'])==0) or 
                print('embedding', index)
                try:
                    arr = pd.DataFrame(row).T.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
                    print('arr', len(arr.iloc[0])) #type(arr.iloc[0]))
                    df.at[index, 'embeddings'] = '[' + ','.join(str(f) for f in arr.iloc[0]) + ']' #np.array2string(np.array(arr.iloc[0]), separator=",")
                except Exception  as e:
                    print('error', e)
                    df.at[index, 'embeddings'] = np.array2string((np.full((1536), -100.0)), separator=",") # [','.join([-1.0]*df.at[index, 'n_tokens'])] * 1 #",".join(np.full((df.at[index, 'n_tokens']), -1.0))    
                time.sleep(0.1) #time.sleep(1.1)
        else:
            df.at[index, 'embeddings'] = np.array2string((np.full((1536), -1.0)), separator=",") # [','.join([-1.0]*df.at[index, 'n_tokens'])] * 1 #",".join(np.full((df.at[index, 'n_tokens']), -1.0))    
    
    print(df)

    df.to_csv('processed/embeddings_ku.csv')
    #df.to_pickle(path='processed/embeddings_ku.pkl', compression='gzip')

    df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
    return render_template('index.html', page=1, web_data = urls, html_data = '', web_data_len = len(urls),  temp='', file = '')   

    #df.head()

def embedding_checknull(df):
#    global df
    global np
    #print(df.head())
 #   if 'embeddings' not in df:
 #       df['embeddings'] = ''


    #print()
    #df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)


    for index, row in df.iterrows():
    #    print(index, df.at[index, 'embeddings']) # df.iloc[[index]]
    #    arr = pd.DataFrame(row).T.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
    #    print('arr', arr.iloc[0])
        #str(df.at[index, 'embeddings']).fillna('', inplace=True)
  #      print(str(df.at[index, 'embeddings']))
        if (not hasattr((df.at[index, 'embeddings']), "__len__")):
            print('embedding', index)
            try:
                arr = pd.DataFrame(row).T.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
                #print('arr', len(arr.iloc[0]))
                df.at[index, 'embeddings'] = arr.iloc[0]

                #df.at[index, 'embeddings'] = df.at[index, 'embeddings'].apply(eval).apply(np.array)
            except Exception  as e:
                print('error', e)
                df.at[index, 'embeddings'] = np.array2string((np.full((1536), -100.0)), separator=",") # [','.join([-1.0]*df.at[index, 'n_tokens'])] * 1 #",".join(np.full((df.at[index, 'n_tokens']), -1.0))    
            time.sleep(0.1) #time.sleep(1.1)

    #df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)
    # df['embeddings'] = df['embeddings'].apply(np.array)
    #return render_template('index.html', page=1, web_data = urls, html_data = '', web_data_len = len(urls),  temp='', file = '')   
# reinit DF from file


#df=pd.read_csv('processed/embeddings.csv', index_col=0)
#df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

def create_context(question, df, max_len=2800, size="ada", isNewEmbed=False):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    if not isNewEmbed:
        q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']
    else:
        q_embeddings = model.encode(question, convert_to_tensor=False) #openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']  
        test_embded = model.encode('кнопка-цвях 36 шт кольорова economix e41102', convert_to_tensor=False)
        test_embded2 = model.encode('добриво органічне концентрат гумінових кислот для кімнатних квітучих рослин 500мл', convert_to_tensor=False)
        print('test_cosine', util.cos_sim(test_embded, q_embeddings), util.cos_sim(q_embeddings, test_embded))
        print('test_cosine2', util.cos_sim(test_embded2, q_embeddings), util.cos_sim(q_embeddings, test_embded2))
        q_embeddings.shape
        q_embeddings_all = df['embeddings2']  
        q_embeddings_all.shape

    #print(type(q_embeddings), type(df.iloc[0]['embeddings']), len(q_embeddings), len(df.iloc[0]['embeddings']))
    #print(['q_embeddings', q_embeddings])
    print(['Create_Context: ', question])
    if not isNewEmbed:
        q_embeddings = np.array(q_embeddings)


    #print(isNewEmbed, question, type(q_embeddings), df.dtypes)
    # Get the distances from the embeddings
    if not isNewEmbed:
        df['dist'] = df.apply(lambda row: Euclidean_Dist(q_embeddings, row['embeddings'], row['text']), axis=1)
    else:
        #df['dist'] = df.apply(lambda row: Euclidean_Dist(q_embeddings, row['embeddings2'], row['text']), axis=1)
        cosine_scores = util.cos_sim(q_embeddings_all, q_embeddings)
        df['dist'] = cosine_scores
        #print(cosine_scores)
        #df['dist'] = df.apply(lambda row: util.cos_sim(q_embeddings, row['embeddings2']), axis=1)
        #util.cos_sim(embedding, embedding2)

    df['distances'] = df['dist'] #distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')
    #df['dist'] = dist(q_embeddings, df['embeddings'].values)

    #print(["distances", df.sort_values('distances', ascending=True)])
    #d_sorted = df.sort_values(by=['dist'], ascending=False) #dict(sorted(df.items(), key=lambda x: x[8], reverse=True))
    #print(d_sorted)

    #print(["dist", df.sort_values('dist', ascending=True)])

    returns = []
    retrows = []
    cur_len = 0

    #print(df)
    # Sort by distance and add the text to the context until the context is too long
    if isNewEmbed == False:
        for i, row in df.sort_values('distances', ascending=True).iterrows():

            #print([row.text, row.distances, row.fname])

            # Add the length of the text to the current length
            cur_len += row['n_tokens'] + 4

            # If the context is too long, break
            if cur_len > max_len:
                break

            #print(row)
            #print(i - 1, cur_len, row["text"])     
            # Else add it to the text that is being returned
            #returns.append(row["text_orig"])
            #print(row["text_orig"])
            if str(row["text_orig"]) != 'nan':
                returns.append(str(row["text_orig"]))
            else:   
                returns.append(str(row["text"])) 
            retrows.append(row)
    else:
        for i, row in df.sort_values('distances', ascending=False).iterrows():

            #print([row.text, row.distances, row.fname])

            # Add the length of the text to the current length
            cur_len += row['n_tokens'] + 4

            # If the context is too long, break
            if cur_len > max_len:
                break

            #print(row)
            #print(i - 1, cur_len, row["text"])     
            # Else add it to the text that is being returned
            #returns.append(row["text_orig"])
            #print(row["text_orig"])
            if str(row["text_orig"]) != 'nan':
                returns.append(str(row["text_orig"]))
            else:   
                returns.append(str(row["text"])) 
            retrows.append(row)        
    #print('!!!', cur_len, len(returns[0]), returns)
    # Return the context
    #print('retrows', returns)
    return ["\n\n###\n\n".join(returns), retrows]

def ask_chat(    
        model="gpt-3.5-turbo-instruct", #"text-davinci-003",
        question="",
        max_len=1800,
        size="ada",
        debug=False,
        max_tokens=2048,#150,
        stop_sequence=None,
        with_leaves = False):
    try:
        # Create a completions using the question and context
        # Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext

        response = openai.Completion.create(
            prompt=f"Напиши український синоним з канцелярії для '{question}'",
            temperature=0.1,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        #print('resp', response["choices"][0]["text"].strip())
        #ret[0][0] = response["choices"][0]["text"].strip()

        return response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)

def answer_question(
    df,
    model="gpt-3.5-turbo-instruct", #"text-davinci-003",
    question="Какие существуют варианты работы с процессами?",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=2048,#150,
    stop_sequence=None,
    with_leaves = False,
    isNewEmbed = False
):
    global urlsindb
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    ret_context = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
        isNewEmbed=isNewEmbed
    )
    print('ret_context:', len(ret_context[1]))
    #print('ret_context:', ret_context[1])
    whole_context = ''
    ret = []
    used_indices = []
    n_tokens = 0
    #df.set_index("Name", inplace = True)
    for c in ret_context[1]:
            currenturl = ''
            #if c['text_orig'].isnumeric():
            #print('NUMBER', c['text_orig'], c['art'], type(c['text_orig']))
            if isinstance(c['text_orig'], str):
                whole_context = whole_context + str(c['text_orig'])
                #print('NUMBER', str(c['text_orig']) )
            else:    
                whole_context = whole_context + str(c['text']) 
                #print('NUMBER2', str(c['text']) )
            ret.append([whole_context, currenturl, str(n_tokens), c['art'], c['dist'], c['qty_leaves'], c['price_spec']])
            whole_context = ''
            n_tokens = 0

    #whole_context = whole_context.replace("\r","").replace("\n","").replace("\r\n", "").replace("¶", '')
    context = ret_context[0]
    #debug = False
 #   print('whole_context', whole_context)
 #   print('context\n\n', ret_context[1])
    # If debug, print the raw model response
    
    return ret

    if debug:
        print("Context:\n" + context)
        print("\n\n")

    #return  ret #[whole_context, currenturl]
    context = ret[0][0]
    print('len', len(context))

    try:
        # Create a completions using the question and context
        # Answer the question based on the context below, and if the question can't be answered based on the context, say \"I don't know\"\n\nContext

        response = openai.Completion.create(
            prompt=f"Используй контекст. Если невозможно ответить, то скажи \"Я не знаю\"\n\n Context: {context}\n\n---\n\nQuestion: {question}\nAnswer:",
            temperature=0,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
            model=model,
        )
        print('resp', response)
        ret[0][0] = response["choices"][0]["text"].strip()

        return ret #response["choices"][0]["text"].strip()
    except Exception as e:
        print(e)
        return ""

def Euclidean_Dist(df1, df2, text):
    try:
        return np.linalg.norm(df1 - df2)
    except (ValueError, TypeError):
        pass
        #print('EuclError')
        #print([ValueError, TypeError, text])
    
#Start reading
i = 0
# For Left join
if os.path.exists('processed/scraped_ku.csv'):
    df = pd.read_csv('processed/scraped_ku.csv', index_col=0)

print(datetime.datetime.now())
force_checking = False
if os.path.exists('processed/embeddings_ku.pkl') and not force_checking: # and not force_checking
    print('Read pkl')
    df_merged = pd.read_pickle('processed/embeddings_ku.pkl', 'gzip')
    df = df_merged
    if 'n_tokens_y' in df and not 'n_tokens' in df:
        df['n_tokens'] = df['n_tokens_y'] 
        df = df.drop(columns=['n_tokens_x', 'n_tokens_y'])

    #print(df)
    #df.to_csv('processed/embeddings_ku2.csv')
#    print(df)    
#    df_merged['embeddings'] = df_merged['embeddings'].apply(eval).apply(np.array)
else:	
    # check existsing rows without vectors
    print('Read csv')
    if os.path.exists('processed/embeddings_ku.csv'):
        df_embed = pd.read_csv('processed/embeddings_ku.csv', index_col=0)
        if (False):
            df_embed["text_orig"] = df_embed['text']#.apply(lambda x: x.lower().replace('"', " ").strip())
            df_embed["text"] = df_embed['text'].apply(lambda x: x.lower().replace('"', " ").strip())
            df_merged = pd.merge(df, df_embed, on=['text', 'text'], how='left')
            df = df_merged	
            if 'n_tokens_y' in df and not 'n_tokens' in df:
                df['n_tokens'] = df['n_tokens_y'] 
                df = df.drop(columns=['n_tokens_x', 'n_tokens_y'])	
        else:
            df = df_embed        

        for index, row in df.iterrows():
            if len(row) > 0 :
                #print(index, df.at[index, 'text'])
                #print(index, df.at[index, 'text'], str(df.at[index, 'embeddings'])[1:4]) # eval(df.at[index, 'embeddings'])[0] #, type(df.at[index, 'embeddings'])) # , eval(row.embeddings).apply(np.array)[0]
                if (str(df.at[index, 'embeddings'])[1:6] == '-100.'):
                    print('embedding_start', index, df.at[index, 'text'])
                    #arr = openai.Embedding.create(input=df.loc[i2, "text"], engine='text-embedding-ada-002')['data'][0]['embedding']
                    arr = pd.DataFrame(row).T.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
                    #q_embeddings = openai.Embedding.create(input=question, engine='text-embedding-ada-002')['data'][0]['embedding']
                    #print(arr)
                    df.at[index, 'embeddings'] = '[' + ','.join(str(f) for f in arr.iloc[0]) + ']' #np.array2string(np.array(arr.iloc[0]), separator=",") #arr.iloc[0]
                    df.at[index, 'embeddings'] = eval(df.at[index, 'embeddings'])
                    #print(df.at[index, 'embeddings'])
                    i+=1
                else:
                    #df.at[index, 'embeddings'] = eval(df.at[index, 'embeddings']) #df['embeddings'].apply(eval)
                    pass
                    #df.at[index, 'embeddings'] = np.array(df.at[index, 'embeddings'])#.apply(eval).apply(np.array)
                    #df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

        df['embeddings'] = df['embeddings'].apply(eval)

        if (i > 0):
            df.to_csv('processed/embeddings_ku.csv') 

#print(df)
#print(['null', len(df[df.embeddings[0] == -1])])
print(datetime.datetime.now())

replace_pkl = False

# check or add lowercase names
if not 'text_orig' in df and False:
    df["text_orig"] = df['text']#.apply(lambda x: x.lower().replace('"', " ").strip())
    df["text"] = df['text'].apply(lambda x: x.lower().replace('"', " ").strip())
    replace_pkl = True
    df.to_csv('processed/embeddings_ku.csv') 


#df = df[['art', 'text','text_orig','n_tokens','embeddings']] 
#df.to_csv('processed/embeddings_ku.csv')


#replace_pkl = True
# try to save
if not os.path.exists('processed/embeddings_ku.pkl') or replace_pkl:
    df.to_pickle(path='processed/embeddings_ku.pkl', compression='gzip')

#print('df0', df)
if 'qty_leaves' in df:
    df = df.drop(columns=['qty_leaves'])

if 'price_spec' in df:
    df = df.drop(columns=['price_spec'])

#get leaves
# 
df_leaves = pd.DataFrame()
if os.path.exists('processed/leaves/leaves_ku.csv'):
    df_leaves = pd.read_csv('processed/leaves/leaves_ku.csv', sep=';', index_col=False)
    df_leaves['art_leaves'] = 'g' + ('000000' + df_leaves['art_leaves'].astype(str)).str[-6:]

df = pd.merge(df, df_leaves, left_on='art', right_on='art_leaves', how='left')#df.merge(df_leaves, left_on='art', right_on='art_leaves', validate='one_to_one')#pd.merge(df, df_leaves, on=['art', 'art_leaves'], how='left')#df.merge(df_leaves, on=['art_leaves', 'art'], how='left')
df = df.drop(columns=['name_leaves', 'code_leaves', 'art_leaves'])	


#get specs        
if os.path.exists('processed/leaves/spec_ku.csv'):
    df_spec = pd.read_csv('processed/leaves/spec_ku.csv', sep=';', index_col=False)
    df_spec['art_spec'] = 'g' + ('000000' + df_spec['art_spec'].astype(str)).str[-6:]

df_spec = df_spec.drop(columns=['№_spec', 'code_spec', 'name_spec'])	

df = pd.merge(df, df_spec, left_on='art', right_on='art_spec', how='left')#df.merge(df_leaves, left_on='art', right_on='art_leaves', validate='one_to_one')#pd.merge(df, df_leaves, on=['art', 'art_leaves'], how='left')#df.merge(df_leaves, on=['art_leaves', 'art'], how='left')
#df = df.drop(columns=['№_spec', 'code_spec', 'art_spec', 'name_spec'])	
df = df.drop(columns=['art_spec'])

df['qty_leaves'] = df['qty_leaves'].fillna(0)
df['price_spec'] = df['price_spec'].fillna(0)

df_leaves = df[df["qty_leaves"] > 0].copy()
df_spec = df[df["price_spec"] != 0].copy()

if False:
    print('calc embeddings2')
    # df['column'] = None
    df["embeddings2"] = None
    cnt = 0
    for index, row in df.iterrows():
        cnt+=1
        df.at[index, 'embeddings2'] = model.encode(df.at[index, 'text'], convert_to_tensor=False)
        if cnt % 50 == 0:
            print(cnt)
        #if cnt == 500:
        #    break    
        #embedding = model.encode(sentences, convert_to_tensor=False)
    #embedding.shape
    if not os.path.exists('processed/embeddings_ku2.pkl') or replace_pkl:
        df.to_pickle(path='processed/embeddings_ku2.pkl', compression='gzip')

print(df)   

if os.path.exists('./processed/synonyms/synonyms_sys.xlsx'):
    df_file = pd.read_excel('./processed/synonyms/synonyms_sys.xlsx', header=None, skiprows=[0])
    dict = parse_lines_synonym(df_file) 
    for filename in os.listdir('./processed/synonyms/'):
        #f = os.path.join(directory, filename)
        # checking if it is a file
        #if os.path.isfile(f):
        if filename != 'synonyms_sys.xlsx' and '.xlsx' in filename:
            print(filename)
            df_file = pd.read_excel('./processed/synonyms/' + filename, header=None, skiprows=[0])
            dict = parse_lines_synonym(df_file) 

#    df_file = pd.read_excel('./processed/synonyms/synonyms_sys.xlsx', header=None, skiprows=[0])
#    dict = parse_lines_synonym(df_file) 

#dict_synonym_terms = {}
#dict_synonym_values = {}
#print('syn_t:', dict_synonym_terms)
print('syn_v:', dict_synonym_values)
#print('syn_v_a:', dict_synonym_values_adds)
#print('syn_v_s:', dict_synonym_values_sorted)

#df_spec["price_spec"] = df_spec["price_spec"].values[0]
#df_spec["price_spec"] = df_spec["price_spec"].astype(str).replace(",", ".") #str(df_spec["price_spec"]).replace(",", ".")
#df_spec["price_spec"] = float(df_spec["price_spec"].replace(",", "."))
#df['price_spec'] = locale.atof(str(df['price_spec']).replace(",", "."))
#df_spec = df[df["price_spec"] > 0].copy()
#print(df[df["price_spec"] != 0])
#print(df.dtypes)
#print(df_spec.dtypes)
#print(type(df_spec["price_spec"].values[0]))
#print(df_spec)
#print(df)
#print(df_leaves)
#print(df_spec)
#print(df)
#df.to_csv('processed/leaves/leaves_ku2.csv') 
#df = pd.read_pickle('processed/embeddings_ku.pkl', 'gzip')

#print(df)

if __name__ == '__main__':
	app.run(debug=True)
	
	#print('Run!')      



