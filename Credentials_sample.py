"""
This file sets the credentials to run the tests and other scripts which require access to Watson Assistant.
"""

# Adoptions (= instances) - name each of these for ease of reference.
adoption1 = ""
adoption2 = ""
adoption3 = ""

# Pick the adoption you want the credentials file to point to
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

# Function to calculate workspace threshold.
# Edit the contents of the if function to change workspace thresholds.
def calculate_workspace_thresh(topic):
    if topic == 'master':
        workspace_thresh = 0.75
    elif topic == 'exception-handling':
        workspace_thresh = 0.8
    else:
        workspace_thresh = 0.4

    return workspace_thresh
