NO=input('Key the number:');
c=gTruth.ROILabelData.Environment.lane; % Import the location of your lane markers data here and run this code
N=length(c{NO,1});
conclu={};
disp(length(c{NO,1}))
for i=1:1:length(c{NO,1})
    A1=c{NO,1}{i,1};
    A1(:, [2, 1]) = A1(:, [1, 2]);
    conclu{i}=A1;
end
n = length(conclu);
for i = 1:n
    eval(sprintf('conc%d = conclu{%d};', i, i));
end
for i=1:n
    dlmwrite('AA.txt','LANE','-append')
    dlmwrite('AA.txt',conclu{i},'-append')
end
if n==0
    dlmwrite('NILL.txt','NILL','-append')
end