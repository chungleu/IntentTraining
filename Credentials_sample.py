"""
This file sets the credentials to run the tests and other scripts which require access to Watson Assistant.
"""

# Adoptions
adoption1 = ""
adoption2 = ""
adoption3 = ""

# Requires user input, depending on which adoption code runs for
active_adoption = adoption1

# Service credentials - can use apikey or username/password
ctx = {
  adoption1: {
    "url": "https://gateway-fra.watsonplatform.net/conversation/api",
    "apikey": ""
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
    "workspace-name": "workspace-id",
    "workspace2": ""
  },
  adoption2: {
    "workspace3": ""
  },
  adoption3: {
    "workspace4": ""
  }
}

# conversation version
conversation_version = '2018-07-10'

# Function to calculate workspace threshold
def calculate_workspace_thresh(topic):
    """
    Simple if/else to return a workspace threshold for a given topic. 
    Edit the below functions to change workspace confidence thresholds.
    """
    if topic == 'master':
        workspace_thresh = 0.75
    elif topic == 'exception-handling':
        workspace_thresh = 0.8
    else:
        workspace_thresh = 0.4

    return workspace_thresh
