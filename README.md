# relative-density
relative density (RD) method implementation
## 参数说明
1.训练集的标签属于{+1，-1} </br>
2.k近邻数为9，阈值设为1   </br>
3.返回“干净”训练集  </br>
## 使用指南
new_train = relative_density(train, k=9, d=1)</br>
accuracy = kernel_SVM(new_train, test)</br>

