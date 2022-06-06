# SRS-WebRTC
**```Introdution```**: 实现游客自由登录网页时，获取本地相机权限，通过WebRTC推流到云端的SRS服务器上，由SRS分流调控。本地运行python程序并捕获视频流抽帧处理。最后通过RTMP推流到服务器，并交由服务器重新分流到游客访问页面

## 一. srs服务器搭建
**参考官网地址**：https://github.com/ossrs/srs/wiki/v4_CN_Home#srs-overview
不看官网wiki必翻车！   
**bilbil视频**：https://www.bilibili.com/video/BV1Lh411v7kD/?p=2&spm_id_from=pageDriver  
**SRS控制台入口**：http://47.112.130.31:8080/console/ng_index.html#/connect  
**测试WebRTC入口**：http://47.112.130.31:8080/players/srs_player.html

## 二.基本命令
**1.查看srs服务器进程**
```angular2html
ps -ef | grep srs
```

**2.启动srs服务器**
```angular2html
./objs/srs -c conf/srs.conf
```

**3.查看SRS的状态**
```angular2html
./etc/init.d/srs status
```


**4.看SRS的日志**
```angular2html
tail -n 30 -f ./objs/srs.log
```

## 三.测试WebRTC拉流
**1.首先开通防火墙端口**  
```Note```：注意有坑！从控制台开启防火墙端口是内网端口，外网还是没有开。所以需要手动用firewall-cmd命令开启

**2.firewall-cmd命令**
1)查看所有打开的端口：
```angular2html
 firewall-cmd --list-all
```
2)将端口1935， 8080， 1985， 8000分别打开，注意：端口8000的协议类型为udp，其他端口协议类型为tcp：
```angular2html
firewall-cmd --add-port=8000/udp --permanent
```
3)重新覆盖：
```angular2html
firewall-cmd --reload
```

**3. 修改rtc2rtmp.conf配置文件**
把其中rtc_server中的candidate改为服务器的公网ip


## 四.测试WebRTC推流
**1.对chrome添加参数
```angular2html
--ignore-certificate-errors --allow-running-insecure-content --unsafely-treat-insecure-origin-as-secure="http://47.112.130.31:8080"
```

## 五.构造页面
由于srs4支持WebRTC，所以提供了一个WebRTC的调试界面并且是开源的，主要用到的是rtc_player.html以及rtc_publisher.html两个文件（srs/trunk/research/players），需要把这两个heml文件以及js，css文件夹拷到本地并整合成我们自己的网页（这样子就可以绕开JavaScript的开发直接使用现成的），修改完还需要对srs服务器重新编译。


## 抓马冷知识 1
问题又来了，由于WebRTC需要https ssl证书才能获取相机权限（避免游客信息泄露），HTTPS就是一个加密证书，由互联网机构签发。
HTTPS需要域名，是因为个人不能够申请指向IP的SSL证书
这些机构能够给域名或者IP签发SSL证书。
IP只能给企业签发，并且申请人必须对IP拥有所有权（我们买的ECS那些IP，相当于从阿里云租来的）
然后，域名如果解析指向中国境内的服务器，就必须备案（为了了解该网站的用途以及开发者是谁）
如果被抓到没备案，就跟这幅图一样，会被自动拦截访问  
![image](https://github.com/Daming-TF/SRS-WebRTC/blob/master/image/%E6%8A%93%E9%A9%AC%E5%86%B7%E7%9F%A5%E8%AF%861.png)

## 抓马冷知识 2
如何架一台境外服务器，比较实惠的就是购买腾讯云偏远地区的服务器，架一个宝塔系统。进入控制面板需要重置密码以及绑定密匙。之后就是安装一些默认需要的工具

进入面板->网络->PHP服务，增加站点填写域名，之后就进入根目录，把html改名为index.html，并且把js，css文件夹放进去根目录下即可访问（srs.acetaffy.club）  
![image](https://github.com/Daming-TF/SRS-WebRTC/blob/master/image/%E6%8A%93%E9%A9%AC%E5%86%B7%E7%9F%A5%E8%AF%862.png)