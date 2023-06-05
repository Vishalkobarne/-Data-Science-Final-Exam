#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install selenium


# In[2]:


pip install msedge-selenium-tools


# In[3]:


# Installing required packages
import csv
from getpass import getpass
from time import sleep
import selenium.webdriver.common.keys
from selenium.common.exceptions import NoSuchElementException
from msedge.selenium_tools import Edge, EdgeOptions


# In[4]:


op = EdgeOptions()
op.use_chromium = True
dr = Edge(options=op)


# In[5]:


dr.get('https://www.twitter.com/login')


# In[6]:


uname = dr.find_element_by_xpath('//input[@name="session[username_or_email]"]')


# In[7]:


# you will have to give your email id here
uname.send_keys('vishalkobarne0405@gmail.com')


# In[8]:


my_password = getpass()


# In[9]:


pwd = dr.find_element_by_xpath('//input[@name="session[password]"]')
pwd.send_keys(my_password)


# In[10]:


# you will have to give your password here
pwd.send_keys(selenium.webdriver.common.keys.Keys.RETURN)


# In[11]:


cds = dr.find_elements_by_xpath('//div[@data-testid="tweet"]')


# In[12]:


cd = cds[2]


# In[13]:


# username

cd.find_element_by_xpath('./div[2]/div[1]//span').text


# In[14]:


# twitter handle
cd.find_element_by_xpath('.//span[contains(text(),"@")]').text


# In[15]:


# content of tweet
content = cd.find_element_by_xpath('.//div[2]/div[2]/div[1]').text
reponding = cd.find_element_by_xpath('.//div[2]/div[2]/div[2]').text
content + reponding


# In[16]:


content


# In[17]:


reponding


# In[18]:


# reply count
cd.find_element_by_xpath('.//div[@data-testid="reply"]').text


# In[19]:


# retweet count
cd.find_element_by_xpath('.//div[@data-testid="retweet"]').text


# In[20]:


# retweet count
cd.find_element_by_xpath('.//div[@data-testid="retweet"]').text


# In[21]:


'''extracting data from tweet data'''
def get_tweet_data(cd):
    
    uname = cd.find_element_by_xpath('./div[2]/div[1]//span').text
    handle = cd.find_element_by_xpath('.//span[contains(text(),"@")]').text
    content = cd.find_element_by_xpath('.//div[2]/div[2]/div[1]').text
    reponding = cd.find_element_by_xpath('.//div[2]/div[2]/div[2]').text
    text = content + reponding
    reply = cd.find_element_by_xpath('.//div[@data-testid="reply"]').text
    retweet = cd.find_element_by_xpath('.//div[@data-testid="retweet"]').text
    like = cd.find_element_by_xpath('.//div[@data-testid="like"]').text
    tweet = (uname, handle, text, reply, retweet, like)
    return tweet


# In[22]:


get_tweet_data(cd)


# In[23]:


tweet_data = []
for cd in cds:
    data = get_tweet_data(cd)
    if data:
        tweet_data.append(data)


# In[24]:


tweet_data[0]


# In[25]:


dr.execute_script("window.scrollTo(0,document.body.scrollHeight);")


# In[26]:


data=[]
tweet_ids=set()
last_position = dr.execute_script("return window.pageYOffset;")
scrolling = True


# In[27]:


while True:
    page_cards = dr.find_elements_by_xpath('//div[@data-testid="tweet"]')
    for cd in page_cards[-15:]:
        tweet = get_tweet_data(cd)
        if tweet:
            tweet_id = "".join(tweet)
            if tweet_id not in tweet_ids:
                tweet_ids.add(tweet_id)
                data.append(tweet)
                
    scroll_attempt=0
    while True:
        dr.execute_script("window.scrollTo(0,document.body.scrollHeight);")
        sleep(1)
        curr_position = dr.execute_script("return window.pageYOffset;")
        if last_position == curr_position:
            scroll_attempt +=1
            if scroll_attempt >= 3:
                scrolling = False
                break
            else:
                sleep(3)
        else:
            last_position = curr_position
            break


# In[28]:


len(data)


# In[29]:


with open('Twitter_Trending.csv','w',newline='',encoding='utf-8') as f:
    header = ['username', 'handle', 'text', 'reply', 'retweet', 'like']
    writer=csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)


# In[30]:


import pandas as pd
df = pd.DataFrame(data,columns=['uname', 'handle', 'text', 'reply', 'retweet', 'like'])


# In[31]:


## This will give our top 10 rows.

df.head(10)


# In[ ]:





# In[ ]:





# In[ ]:




