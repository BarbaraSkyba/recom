import os
import datetime
import pandas as pd
import openai
import numpy as np

def premain(with_relink = False, replace_pkl = False, force_checking = False): 
       
    #Start reading
    i = 0
    # For Left join
    if os.path.exists('processed/scraped_ku.csv'):
        df = pd.read_csv('processed/scraped_ku.csv', index_col=0)

    print(datetime.datetime.now())

#READ PKL
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

            for index, row in df.loc[df['embeddings'].str[0:13] == '[-100.,-100.,'].iterrows():
                if len(row) > 0 :
                    #print(index, df.at[index, 'text'])
                    #print(index, df.at[index, 'text'], str(df.at[index, 'embeddings'])[1:4]) # eval(df.at[index, 'embeddings'])[0] #, type(df.at[index, 'embeddings'])) # , eval(row.embeddings).apply(np.array)[0]
                    if (str(df.at[index, 'embeddings'])[0:13] == '[-100.,-100.,'): # [1:6] == '-100.'
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
                if index % 50 == 0:
                    print('check row', index)

            df['embeddings'] = df['embeddings'].apply(eval)
            print('updated rows:', i)
            #if (i > 0):
            #    df.to_csv('processed/embeddings_ku.csv') 

    #print(df)
    #print(['null', len(df[df.embeddings[0] == -1])])
    print(datetime.datetime.now())

    if with_relink:
#   RELINK NEW FILES
        relink_new_ent_list(df, 'processed/embeddings_ku_240425.csv')

    

    # check or add lowercase names
    if not 'text_orig' in df and False:
        df["text_orig"] = df['text']#.apply(lambda x: x.lower().replace('"', " ").strip())
        df["text"] = df['text'].apply(lambda x: x.lower().replace('"', " ").strip())
        replace_pkl = True
        df.to_csv('processed/embeddings_ku.csv') 


    #df = df[['art', 'text','text_orig','n_tokens','embeddings']] 
    #df.to_csv('processed/embeddings_ku.csv')

    # try to save PKL
    if not os.path.exists('processed/embeddings_ku.pkl') or replace_pkl:
        df.to_pickle(path='processed/embeddings_ku.pkl', compression='gzip')


#LEAVES
    if 'qty_leaves' in df:
        df = df.drop(columns=['qty_leaves'])

#SPEC
    if 'price_spec' in df:
        df = df.drop(columns=['price_spec'])

#GET LEAVES
    df_leaves = pd.DataFrame()
    df['qty_leaves'] = 0.0
    if os.path.exists('processed/leaves/leaves_all_ku.csv'):
        df_leaves = pd.read_csv('processed/leaves/leaves_all_ku.csv', sep=';', index_col=False)
        df_leaves['leaves_qty_l'] = df_leaves['leaves_qty_l'].str.replace(',', '.').str.replace(' ', '') #float(df_leaves['leaves_qty_l'])
        if 'art_leaves' in df_leaves:
            df_leaves['art_leaves'] = 'g' + ('000000' + df_leaves['art_leaves'].astype(str)).str[-6:]

    if 'art_leaves' in df_leaves:
        df = pd.merge(df, df_leaves, left_on='art', right_on='art_leaves', how='left')#df.merge(df_leaves, left_on='art', right_on='art_leaves', validate='one_to_one')#pd.merge(df, df_leaves, on=['art', 'art_leaves'], how='left')#df.merge(df_leaves, on=['art_leaves', 'art'], how='left')
        df = df.drop(columns=['name_leaves', 'code_leaves', 'art_leaves'])	


#GET SPECS        
    df_spec = pd.DataFrame()
    df['price_spec'] = 0.0
    """
    if os.path.exists('processed/leaves/spec_ku.csv'):
        df_spec = pd.read_csv('processed/leaves/spec_ku.csv', sep=';', index_col=False)
        df_spec['art_spec'] = 'g' + ('000000' + df_spec['art_spec'].astype(str)).str[-6:]

    df_spec = df_spec.drop(columns=['№_spec', 'code_spec', 'name_spec'])	

    df = pd.merge(df, df_spec, left_on='art', right_on='art_spec', how='left')#df.merge(df_leaves, left_on='art', right_on='art_leaves', validate='one_to_one')#pd.merge(df, df_leaves, on=['art', 'art_leaves'], how='left')#df.merge(df_leaves, on=['art_leaves', 'art'], how='left')
    #df = df.drop(columns=['№_spec', 'code_spec', 'art_spec', 'name_spec'])	
    df = df.drop(columns=['art_spec'])
    """
    # name_agent_s;code_agent_s;name_dict_s;code_dict_s;code_agent_s_n;code_dict_s_n    
    if os.path.exists('processed/leaves/spec_all_ku.csv'):
        df_spec = pd.read_csv('processed/leaves/spec_all_ku.csv', sep=';', index_col=False)
        #df['price_spec'] = 1
        #df_spec = pd.merge(df_spec, df, left_on='code_n2', right_on='code_n2', how='left')
        if 'art_spec' in df_spec:
            df_spec['art_spec'] = 'g' + ('000000' + df_spec['art_spec'].astype(str)).str[-6:]    

#GET SALES        
    df_sales = pd.DataFrame()

 
    if os.path.exists('processed/leaves/sales_all_ku.csv'):
        df_sales = pd.read_csv('processed/leaves/sales_all_ku.csv', sep=';', index_col=False)

    wrk = df.set_index('code_n2')
    #print('wrk', wrk)
    #print('wrkk', df_leaves.set_index('code_n2'))
    if 'leaves_qty_l' in df_leaves:
        #df['qty_leaves'] = np.where(df['code_n2'] == df_leaves['code_n2'], df['qty_leaves'], df_leaves['leaves_qty_l'])
        wrk.qty_leaves.update(df_leaves.set_index('code_n2').leaves_qty_l.apply(float))
        #print('wrk0', wrk[wrk.index.isin(['g000048552'])]) # wrk[wrk['code_n2']=='g000048552']
        #print(df_leaves)
        #print('wrk0', df_leaves[df_leaves['code_n2']=='g000048552'])



    #if 'code_dict_s' in df_spec:
        #df['price_spec'] = np.where(df['code_n2'] == df_spec['code_n2'], df['price_spec'], df_spec['code_dict_s'])
    #    wrk.price_spec.update(df_spec.set_index('code_n2').code_dict_s)

    df = wrk.reset_index().reindex(columns=list(df.columns.values))


    df['qty_leaves'] = df['qty_leaves'].fillna(0)
    df['price_spec'] = df['price_spec'].fillna(0)

    #print('wrk2', df[df.qty_leaves.apply(type) == 'str'])
    #df['qty_leaves'] = df['qty_leaves'].astype('int') #int(df['qty_leaves'])
    df["qty_leaves"] = pd.to_numeric(df["qty_leaves"], errors='coerce')
    df["price_spec"] = pd.to_numeric(df["price_spec"], errors='coerce')

    #df['price_spec'] = df['price_spec'].astype('int') #int(df['price_spec'])
    #print('wrk', df[df['qty_leaves'].notnull() or df['price_spec'].notnull()])

    df.loc[df.qty_leaves > 0, "qty_leaves"] = 1
   # df.loc[df.price_spec > 0, "price_spec"] = 1

    #print('wrk0', df[df['code_n2']=='g000048552'])
    #print('wrk1', df[df["qty_leaves"] > 0])
    #df['qty_leaves']
    #if 'qty_leaves' in df:


    #print('df_leaves', df_leaves)
    #print('df_spec', df_spec)
    #print('df', df)
      
    if 'qty_leaves' in df:
        df_leaves = df[df["qty_leaves"] > 0].copy()

    if not 'code_a2' in df_spec:
        df_spec['code_a2'] = ''   

    if not 'code_a2' in df_sales:
        df_sales['code_a2'] = ''     
    #if 'price_spec' in df:        
    #    df_spec = df[df["price_spec"] != 0].copy()



    return [df, df_leaves, df_spec, df_sales]

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

    #print(df)   


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

def relink_new_ent_list(df, new_file_name):
    df_embed = pd.read_csv('processed/tmp/Номенклатура вся на 2404.csv', delimiter=';', index_col=False)
    df_merged = pd.merge(df, df_embed, left_on='art', right_on='art_n2', how='outer') #df.merge(df_leaves, left_on='art', right_on='art_leaves', validate='one_to_one')#pd.merge(df, df_leaves, on=['art', 'art_leaves'], how='left')#df.merge(df_leaves, on=['art_leaves', 'art'], how='left')
    df_merged_left = pd.merge(df, df_embed, left_on='art', right_on='art_n2', how='left') #df.merge(df_leaves, left_on='art', right_on='art_leaves', validate='one_to_one')#pd.merge(df, df_leaves, on=['art', 'art_leaves'], how='left')#df.merge(df_leaves, on=['art_leaves', 'art'], how='left')
    
    df_merged_left = df_merged_left.drop(columns=['parentname_n', 'art_n', 'un_n', 'art_n2', 'name_n', 'code_n', 'embeddings2'])
    #df_merged_left['embeddings2'] = []
    #df_merged_left.loc[:,'embeddings2'] = []
    #df_merged_left['embeddings2'] = []
    df_merged_left['embeddings2'] = [list() for x in range(len(df_merged_left.index))]
    df_merged_left = df_merged_left.drop(df_merged_left[(df_merged_left.art.map(len) <= 2)].index)
    print('new_file', df_merged)
    print('new_file_left', df_merged_left)

    df_merged_left.to_csv('processed/embeddings_ku_n.csv') 
    df_merged_left.to_pickle(path='processed/embeddings_ku_n.pkl', compression='gzip')
    pass