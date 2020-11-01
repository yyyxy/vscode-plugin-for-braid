# vscode plugin for braid

A vscode plugin for braid.

## Installation



step1. git clone https://github.com/yyyxy/vscode-plugin-for-braid.git . then `cd vscode-plugin-for-braid`

step2. change directionary to biker/braid, using command `cd biker/braid` 

step3. install the required libraries , using command `pip install -r requirements.txt`

step4. start backend server, using command `python server.py`

![run-server](./assets/images/01-run-server.png)

step5. open plugin-end code folder with [vscode editor](https://code.visualstudio.com/).

<img src="./assets/images/02-open-code-with-vscode.png" alt="open-code-with-vscode" style="zoom: 67%;" />

step6. change to panel `Run`, click the `Start Debugging` button. then vscode will open a new window for you to test the plugin.

<img src="./assets/images/03-debug-panel.png" alt="debug-panel" style="zoom:67%;" />



<img src="./assets/images/04-a-new-vscode-window.png" alt="a-new-vscode-window" style="zoom: 50%;" />

## Usage

step1. `Ctrl+F9` (windows,Linux) or `Cmd+F9` (OSX), and type your query string, for example, "how to sort"

<img src="./assets/images/05-how-to-query.png" alt="how-to-query" style="zoom: 50%;" />

step2. view the query results

<img src="./assets/images/06-query-results.png" alt="query-results.png" style="zoom: 50%;" />

step3. click the desired item to feedback your selection.

<img src="./assets/images/07-click-to-feedback.png" alt="click-to-feedback" style="zoom: 50%;" />