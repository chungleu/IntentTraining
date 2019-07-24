# *****************************************************************
#
# Licensed Materials - Property of IBM
#
# IBM Watson Conversation Test Suite
# (C) Copyright IBM Corp. 2017. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with IBM Corp.
#
# *****************************************************************

##################################################################
# This file is to set the credentials for all the examples to run.

# This should be a copy of your service credentials of the conversation service.

# Adoptions
adoption1 = ""
adoption2 = ""
adoption3 = ""

# Requires user input, depending on which adoption code runs for
active_adoption = adoption2

# Service credentials
ctx = {
  adoption1: {
    "url": "https://gateway-fra.watsonplatform.net/conversation/api",
    "username": "",
    "password": ""
  },
  adoption2: {
    "url": "https://gateway-fra.watsonplatform.net/conversation/api",
    "username": "",
    "password":""
  },
  adoption3: {
    "url": "https://gateway-fra.watsonplatform.net/conversation/api",
    "username":"",
    "password":""
  }
}

# Workspace IDs
workspace_id = {
  adoption1: {
    "workspace1": "",
    "workspace2": ""
  },
  adoption2: {
    "workspace3": ""
  },
  adoption3: {
    "workspace4": ""
  }
}

# Conversation API version to use.
conversation_version = '2017-02-03'

# Alchemy key from your service.
alchemy_key = ''

# Service credentials from your tone analyser service.
tone_ctx = {}
