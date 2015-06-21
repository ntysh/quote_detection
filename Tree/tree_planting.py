#!/usr/bin/python
# -*- coding: utf-8 -*-
quotes = []

endofQuote ='END'

#quoteTree = { 'кто':{'не':{'работает':{'тот':{'__number__':1, '__pos__':'XX', '__'}, '__pos__':'verb'} }}
#            ,'мой':{'Лизочек':{}}
#            } 
#quote = ['кто','не','работает','тот', 'не', 'ест']



import codecs, os, sys, re, csv, pickle, re, types
quoteTree = {}
#f1 = codecs.open("../useful_lists/wings_4grams.txt", 'r', 'utf-8').readlines() #../wings_5grams
#f2 = codecs.open("../useful_lists/wings_3grams.txt", 'r', 'utf-8').readlines()
f = codecs.open("../useful_lists/wings_5grams.txt", 'r', 'utf-8').readlines()

#quotes = [line.rstrip('\n').split('\t') for line in f1]+[line.rstrip('\n').split('\t') for line in f2]  #.encode('utf-8')
quotes = [line.rstrip('\n').split('\t') for line in f]  #.encode('utf-8')

# stats
totalTokens = 0
totalQuotes = 0

def getToken ():
#	"""Берет следующий токен, пока не заканчивается длина разбираемой строки"""
   global quoteIndex,quotePos
   quote = quotes[quoteIndex]
   if quotePos == len(quote) :
#      print 
      return endofQuote
   else :
      word = quote[quotePos]
#      print word,
      quotePos += 1
      return word
     
def addNode(token, Tree) :
#	""" Добавляет в префиксное дерево токены, формируя структуру вида {'мой':{'Лизочек':{'__id__'=0}} """
   global totalTokens, totalQuotes,quoteIndex
   if token == endofQuote :
      Tree['__id__'] = quoteIndex #можно вписывать количественные хар-ки на любом уровне 
      totalQuotes += 1
      return 
   if Tree.get(token) == None :
      Tree[token] = {} 
      totalTokens += 1 
   addNode(getToken(), Tree[token])

def printTree(tree):
   if  type(tree) == types.DictType:
      for k,v in tree.items():
         print k,' ',         
         printTree(v)
   else:
      print tree 
                    
for quoteIndex in range (len(quotes)) :
   quotePos = 0
   addNode(getToken(), quoteTree)  

printTree(quoteTree)


f = codecs.open('quoteTree_5.dict','w','utf-8')
pickle.dump(quoteTree, f) # в какой  файл загружать
f.close()

print 'totalTokens=',totalTokens
print 'totalQuotes=', totalQuotes
#
