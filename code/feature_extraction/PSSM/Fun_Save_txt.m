% Save_Path��.txt�ļ�����·��
% matrix����Ҫ����ı���

function Fun_Save_txt(Save_Path, matrix)
        fid = fopen(Save_Path,'wt');
        [m,n] = size(matrix);
        for i = 1:1:m
            for j = 1:1:n
                if j == n
                    fprintf(fid,'%12.5f\n',matrix(i,j));
                else
                    fprintf(fid,'%12.5f,',matrix(i,j));
                end
            end
        end
        fclose(fid);
end

