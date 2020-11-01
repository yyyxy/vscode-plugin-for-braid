const vscode = require('vscode');
/**
 * 自动提示实现，这里模拟一个很简单的操作
 * 当输入 this.dependencies.xxx时自动把package.json中的依赖带出来
 * 当然这个例子没啥实际意义，仅仅是为了演示如何实现功能
 * @param {*} document 
 * @param {*} position 
 * @param {*} token 
 * @param {*} context 
 */

 var wsUri = "ws://localhost:8765";
 var websocket;
 var socketData;



var test = ['1','2','3']

function initWebSocket(){ //初始化weosocket
    const wsuri = "ws://localhost:8765";
    websocket = new WebSocket(wsuri);
    websocket.onmessage = websocketonmessage;
    websocket.onopen = websocketonopen;
    websocket.onerror = websocketonerror;
    websocket.onclose = websocketclose;
}

function websocketonopen(){ //连接建立之后执行send方法发送数据
    let actions = "this is completion require";
    websocketsend(actions);
}

function websocketonerror(){//连接建立失败重连
    initWebSocket();
}

function websocketonmessage(e){ //数据接收
    var sk = JSON.parse(e.data);
    socketData = sk.codeRecommendation
    console.log("received data:" + sk)
    console.log("json:" + this.socketData)
}

function websocketsend(Data){//数据发送
    websocket.send(Data);
}


function websocketclose(e){  //关闭
    console.log('断开连接',e);
}



function provideCompletionItems(document, position, token, context) {
    const line = document.lineAt(position);
    console.log("completion:" + test)
    return new vscode.CompletionItem(test, vscode.CompletionItemKind.Field);
}

/**
 * 光标选中当前自动补全item时触发动作，一般情况下无需处理
 * @param {*} item 
 * @param {*} token 
 */
function resolveCompletionItem(item, token) {
    return null;
}

module.exports = function(context) {
    // 注册代码建议提示，只有当按下“.”时才触发
    context.subscriptions.push(vscode.languages.registerCompletionItemProvider('javascript', {
        provideCompletionItems,
        resolveCompletionItem
    }, 'alt+q'));
};