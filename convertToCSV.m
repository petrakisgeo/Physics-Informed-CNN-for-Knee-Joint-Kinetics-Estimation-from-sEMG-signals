% Set the working directory
dir_path = pwd;

% Get all .mat files in the current directory and its subfolders
mat_files = dir(fullfile(dir_path, '**/*.mat'));

% Loop through each .mat file
for i = 1:length(mat_files)
    
    % Check if the current directory path containts 'stair', to convert
    % only stair trials
    if ~contains(mat_files(i).folder,'stair','IgnoreCase',true)
        %datastruct=load(mat_files(i));
        continue
    end
    
    % (Condition files handled in Python)
    if contains(mat_files(i).folder,'condition','IgnoreCase',true)
        datastruct=load(fullfile(mat_files(i).folder, mat_files(i).name));
        [~, name, ~] = fileparts(mat_files(i).name);
        %disp(datastruct.speed);
        writetable(table(datastruct.stairHeight,'VariableNames',{'stairHeight'}), ...
            fullfile(mat_files(i).folder, [name '.csv']));
        continue % skip the current iteration and move to the next iteration
    end
    
    % Get the file name without the extension
    [~, name, ~] = fileparts(mat_files(i).name);
    
    % Check if a CSV file with the same name exists
    csv_filename = fullfile(mat_files(i).folder, [name '.csv']);
    if exist(csv_filename, 'file')
        continue % skip the current iteration and move to the next iteration
    end
    
    % Load the .mat file
    mat_data = load(fullfile(mat_files(i).folder, mat_files(i).name));
    
    % Get the file name without the .mat extension
    [~, name, ~] = fileparts(mat_files(i).name);
    
    % Convert the data to a table and write to .csv file
    writetable(mat_data.data, fullfile(mat_files(i).folder, [name '.csv']));
    
end

% % Also convert the subject info file
% data=load('SubjectInfo.mat');
% writetable(data.data,'SubjectInfo.csv');
% disp('Conversion complete!');
