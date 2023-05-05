#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 09:19:07 2020

@author: emma
"""

import numpy as np

# 1. create a list; find ll values < 10

int_list=np.random.randint(0,10,(50,))

int_list_less_than_five=int_list[int_list<5]

# 2. create captital cities dict

cap_cities={}
cap_cities['England']='London'
cap_cities['France']='Paris'
cap_cities['Germany'] ='Berlin'
cap_cities['America']='Washington'
cap_cities['China']='Beijing'

# try iterating over the dictionary and finding all keys that match some condition
for country in cap_cities:
    if country[0]=='G':
        # print the key and object matching this condition
        print('question 2', country, cap_cities[country])
        
        
# 3. Compute factorial

def factorial(num):
    # create a list of all numbers up to and including num
    # remember ranges always are up to one-minus the argument
    fact_list=np.arange(num+1)
    # don't include zero
    fact=fact_list[1]
    #don't
    for i in fact_list[2:]:
        fact*=i
        
    return fact

fact_trial=5
print('question 3: the factorial of {} is {}'.format(fact_trial,factorial(fact_trial)))

# 4. sum of strings

def sum_of_strings(string_one,string_two):
    return int(string_one)+int(string_two)

print(' sum of {} and {} is {}'.format('1','4',sum_of_strings('1','4')))

# 5. rock paper scissors (apologies if you don't know this game)

def rock_paper_scissors(choice):
    # np.random.choice allows for random selection from array or range
    computer_choice=np.random.choice(['rock','paper','scissors'])
    
    if choice == computer_choice:
        print('Draw')
    elif choice=='rock' and computer_choice=='paper':
        print('paper beats rock you lose')
    elif choice=='scissors' and computer_choice=='rock':
        print('rock beats scissors - you lose')
    elif choice=='paper' and computer_choice=='scissors':
        print('scissors beat paper - you lose')
    else:
        print('the computer chose {} so you win!'.format(computer_choice))
        

rock_paper_scissors('rock')


# 6. compare two lists and return identical items
# note this is not the most efficient solution - you can instead use python sets (not covered in course)
# example below for those that are interested
# https://stackoverflow.com/questions/1388818/how-can-i-compare-two-lists-in-python-and-return-matches

def compare_lists(list1,list2):
    
    # first identify the longest list to loop over
    if len(list1) > len(list2):
        longest=list1
        shortest=list2
    else:
        longest=list2
        shortest=list1
    
    common_values=[] # initialise list of common values
    for item in longest:
        if item in shortest:
            common_values.append(item)
    
    return common_values

list1=[1, 2, 3, 4, 5, 6]
list2=[6,7,8,9,2]

compare_lists(list1,list2)