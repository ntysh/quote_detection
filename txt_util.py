#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'tysh'

import codecs, os, sys, re, csv, pickle, types, HTMLParser, nltk


hPrs = HTMLParser.HTMLParser()  # создаётся объект парсера



def cleanHtml(text):
    """Парсит html-текст и чистит его. """

    #	m = re.search(u'<\\!\\-\\-\\s\\=+\\sWITHOUT\\sANY\\sTABLES\\s\\=+\\s\\-\\-\\>(.*?)(в\\sначало\\sстраницы</a>)?<\\!\\-\\-\\s\\=+\\s\\-\\-\\>', f.read(),
    #				  flags=re.U|re.DOTALL)
    m = re.search(u'<\\!\\-\\-\\s\\=+\\sWITHOUT\\sANY\\sTABLES\\s\\=+\\s\\-\\-\\>(.*?)(\\<p\\>_{5,100}\\/p>|<\\/div>)',
                  text,
                  flags=re.U | re.DOTALL)
#    m = re.search(u'<div\sclass\\=\\"content\\">(.*?)<\\/div>',text,flags=re.U | re.DOTALL)
    if m != None:

        text = m.group(1) \
            .replace(u'–', u'–') \
            .replace(u'—', u'–') \
            .replace(u' Ч ', u' – ') \
            .replace(u'...', u'…') \
            .replace(u'┘', u'…') \
            .replace(u'√', u'–') \
            .replace(u'”', u'”') \
            .replace(u'“', u'“') \
            .replace(u'⌠', u'“') \
            .replace(u'■', u'”') \
            .replace(u'≈', u'–') \
            .replace(u'ё', u'е') \
            .replace(u'Ё', u'Е') \
            .replace(u'═', u'@') \
            .replace(u'╚', u'«') \
            .replace(u'╜', u'') \
            .replace(u'á', u'') \
            .replace(u'â', u'') \
            .replace(u'╧', u'') \
            .replace(u'╩', u'»') \
            .replace(u'é', u'') \
            .replace(u'\n', u' ') \
            .replace(u'\r', u'') \
            .replace(u'ó', u'') \
            .replace(u'└', u'“') \
            .replace(u'<i>', u'└') \
            .replace(u'<I>', u'└') \
            .replace(u'</i>', u'┘') \
            .replace(u'</I>', u'┘') \
            .replace(u'<p>', u'≈') \
            .replace(u'<P>', u'≈') \
            .replace(u'</p>', u'≉') \
            .replace(u'</P>', u'≉') \
            .replace(u'&nbsp;', u'■')

        # @ - сноска-единичка
        # └ - открывающий тег курсива
        # ┘ - закрывающий тег курсива
        # ≈ - открывающий тег абзаца
        # ≉ - закрывающий тег абзаца
        # ■ - неразрывный пробел

    return hPrs.unescape(nltk.clean_html(text))  # в строке text все HTML entities заменяются на соответствующие символы
#

def print_with_borders(handle,beginning,ending,answer,id,words):
    handle.write(  str(id)+ ' ' + str(answer) +' ' )
    for j,w in enumerate(words):
        if j == beginning: handle.write('$')
        handle.write(w['text'])
        if j == ending: handle.write('$')
    handle.write('\n')

def write_to_corpora(beginning,ending,answer,words,out_sentences):
    out_sentences.write(str(answer))
    for j in xrange(len(words)):
        if j == beginning: out_sentences.write('|')  # print '|', #
        out_sentences.write(words[j]['text'])  # print words[j]['text'], #
        if j == ending: out_sentences.write('|')  # print '|', #
    out_sentences.write('\n')
