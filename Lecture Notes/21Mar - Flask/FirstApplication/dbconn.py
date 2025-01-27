#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:27:41 2023

@author: sheriffabdullah
"""

import pymysql as pms

conn = pms.connect(host='localhost',
                   port=3306,
                   user='root',
                   password=pas,
                   db='mltlabpractice') #Sundhar@1610

# Can give IP address if server is not the localhost

cur = conn.cursor()
cur.execute("SELECT * FROM employee")
output = cur.fetchall() # fetchone() -> Will bring 1 row @ a time.
print(output)

#%%
# OR (Use Pandas. Advantage = get result as a DataFrame)

import pandas as pd

sql = "SELECT * FROM employee"
pd.read_sql(sql, conn)

#%%

pas = 'myyesqueueyell@Mr.Awesome3040:)'