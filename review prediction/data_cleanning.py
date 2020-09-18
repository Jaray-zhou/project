#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

data_df = pd.read_csv("orders_merge.csv")
data_df.head()


# In[2]:


# drop the null value
data_df = data_df.dropna(subset=["order_status"])
# split the types of order_status
data_df = pd.get_dummies(data_df, columns=["order_status"], prefix=["order_status"])
# calculate whether the sellers and customers come from same state
data_df['same_state'] = np.where(data_df['seller_state'] == data_df['customer_state'], 1, 0)
# calculate whether the scustomers pay by installment
data_df['is_payment_installments'] = np.where(data_df['payment_installments'] >1, 1, 0)


# calculate whether delivery on time or not
data_df["order_delivered_customer_date"] = pd.to_datetime(data_df['order_delivered_customer_date'])
data_df["order_estimated_delivery_date"] = pd.to_datetime(data_df['order_estimated_delivery_date'])
date = (data_df['order_estimated_delivery_date'] - data_df['order_delivered_customer_date']).apply(lambda x: x.days)
date.fillna(date.mean())
data_df["is_delivered_on_time"] = date
data_df["is_delivered_on_time"] = np.where(data_df.is_delivered_on_time >0, 1, 0)

# calculate whether leave a review
data_df["review_score"] = data_df['review_score'].fillna(0)
data_df['review_score'] = np.where(data_df['review_score'] == 0, 0, 1)


# In[3]:


# put the Y in first column
cols = list(data_df)
cols.insert(0,cols.pop(cols.index('review_score')))
data_df= data_df.loc[:,cols]
data_df = data_df.rename(columns={'review_score':'is_review'})


# In[4]:


# drop the features dont need
data_df = data_df.drop(data_df.iloc[:,1:13], axis = 1)
data_df = data_df.drop(data_df.iloc[:,2:6], axis = 1)
data_df = data_df.drop(["product_category_name"], axis = 1)
data_df = data_df.drop(["payment_sequential"], axis = 1)
data_df = data_df.drop(["payment_installments"], axis = 1)
data_df = data_df.drop(["payment_value"], axis = 1)
data_df = data_df.drop(data_df.iloc[:,17:25], axis = 1)
data_df = data_df.drop(["product_category_name_english"], axis = 1)


# In[5]:


# change the datatypes
data_df["order_status_approved"] = data_df["order_status_approved"].astype('int')
data_df["order_status_canceled"] = data_df["order_status_canceled"].astype('int')
data_df["order_status_delivered"] = data_df["order_status_delivered"].astype('int')
data_df["order_status_created"] = data_df["order_status_created"].astype('int')
data_df["order_status_processing"] = data_df["order_status_processing"].astype('int')
data_df["order_status_shipped"] = data_df["order_status_shipped"].astype('int')
data_df["order_status_unavailable"] = data_df["order_status_unavailable"].astype('int')
data_df["order_status_invoiced"] = data_df["order_status_invoiced"].astype('int')


# In[6]:


data_df.dtypes


# In[7]:


data_df = data_df.fillna(round(data_df.mean(),0))


# In[8]:


# export the data to csv
data_df.to_csv("reviews_prediction.csv")


# In[ ]:




