const app = Vue.createApp({
    data() {
        const 所有文件 = ['多标签散点图32×768.html', '多标签散点图32×1024.html', '多标签散点图32×1280.html', '多标签散点图8×1024.html', '模型对标签大类-准确率768.html', '模型对标签类别-准确率768.html', '模型对不同系列的准确率.html', '模型对不同角色的准确率.html', '模型对画师的准确率.html'];
        const query = decodeURIComponent(location.search.slice(1));
        const file = query && 所有文件.find(i => i.includes(query)) || 所有文件[0];
        return {
            所有文件,
            当前文件: './' + file 
        }
    }
})
window.onload = () => app.mount('#app')
