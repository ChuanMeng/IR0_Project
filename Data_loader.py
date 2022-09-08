#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 15:55:41 2022

@author: c.meng
"""

import json

def passage_loader(path):
    print("load passages from: {}".format(path))
    
    passages = json.load(open(path, 'r', encoding="utf-8", errors="ignore"))
    
    return passages

def query_loader(path):
    
    print("load queries from: {}".format(path))
    queries = json.load(open(path, 'r'))
    
    return queries


def label_loader(path):
    print("Load labels from: {}".format(path))
    labels = json.load(open(path, 'r'))
    
    return labels