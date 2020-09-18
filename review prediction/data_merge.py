#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[10]:


# read the data
orders_df = pd.read_csv("olist_orders_dataset.csv")
customers_df = pd.read_csv("olist_customers_dataset.csv")
order_item_df = pd.read_csv("olist_order_items_dataset.csv")
products_df = pd.read_csv("olist_products_dataset.csv")
productnames_df = pd.read_csv("product_category_name_translation.csv")
payment_df = pd.read_csv("olist_order_payments_dataset.csv")
sellers_df = pd.read_csv("olist_sellers_dataset.csv")
reviews_df = pd.read_csv("olist_order_reviews_dataset.csv")


# In[11]:


# merge the dataset of customers and orders
orders_merge_df = orders_df.merge(customers_df, how = "left",  on = "customer_id")

# merge the dataset with order_item
orders_merge_df = orders_merge_df.merge(order_item_df, how = "left",  on = "order_id")

# merge the dataset of product and product name translation
products_df = products_df.merge(productnames_df, how = "left",  on = "product_category_name")

# merge the data with product
orders_merge_df = orders_merge_df.merge(products_df, how = "left",  on = "product_id")

# split the type of payment type
payment_df = pd.get_dummies(payment_df, columns=["payment_type"], prefix=["payment_type"])
# aggregate the data by order_id
payment_group_df = payment_df.groupby(['order_id']).sum()

# merge the data with payment
orders_merge_df = orders_merge_df.merge(payment_group_df, how = "left",  on = "order_id")

# merge the data with sellers
orders_merge_df = orders_merge_df.merge(sellers_df, how = "left",  on = "seller_id")

# merge the data with reviews
orders_merge_df = orders_merge_df.merge(reviews_df, how = "left",  on = "order_id")


# In[ ]:


# check the merged data
orders_merge_df.head()


# In[13]:


orders_merge_df.info()


# In[14]:


# export the data to csv
orders_merge_df.to_csv("orders_merge.csv")


# In[ ]:




