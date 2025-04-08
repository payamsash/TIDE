% T2 functional MRI Analysis Pipeline
% Written by Payam S. Shabestari, Zurich, 01.2025
% email: payam.sadeghishabestari@uzh.ch
% This script is written mainly for Antinomics project. However It could be used for other purposes.

% load functional/structural files
subject_id = 'dvob';
n_subjects = 1;
n_sessions = 2;
TR = 2.5;
subjects_conn_dir = '/home/ubuntu/data/subjects_conn_dir';
subject_dir = fullfile('/home/ubuntu/data/subjects_raw', subject_id);

func_files = cellstr(sort(conn_dir(fullfile(subject_dir, 'fMRI', 'raw_func_s*.nii'))));
struc_file = cellstr(conn_dir(fullfile(subject_dir, 'sMRI', 'raw_anat_T1.nii')));

if length(func_files) ~= 2
    error('Mismatch for subject %s: number of functional files should be 2, but found %d', subject_id, length(func_files));
end

if length(struc_file) ~= 1
    error('Mismatch for subject %s: number of structural files should be 1, but found %d', subject_id, length(struc_file));
end

func_files = reshape(func_files,[n_sessions, n_subjects]);
struc_file = {struc_file{1:n_subjects}};

% run setup and preprocessing

clear batch;
batch.filename = fullfile(subjects_conn_dir, [subject_id, '.mat']);            
batch.Setup.isnew = 1;
batch.Setup.n_subjects = n_subjects;
batch.Setup.RT = TR;
batch.Setup.functionals = repmat({{}}, [n_subjects,1]);

for nsub = 1:n_subjects
    for nses = 1:n_sessions
        batch.Setup.functionals{nsub}{nses}{1} = func_files{nses, nsub}; 
    end
end

batch.Setup.structurals = struc_file;                  
n_conditions=n_sessions;
batch.Setup.conditions.names={'rest', 'rest'};
batch.Setup.preprocessing.steps = 'default_mni';
batch.Setup.preprocessing.sliceorder = 'interleaved (Philips)';
batch.Setup.done = 1;
batch.Setup.overwrite = 'Yes';

% volumetric analysis
% Schaefer atlas
%batch.Setup.rois.files{1}='ROIs/AndyROIs.nii';
%batch.Setup.rois.multiplelabels = 1;


% denoising
% CONN Denoising                                    % Default options (uses White Matter+CSF+realignment+scrubbing+conditions as confound regressors); see conn_batch for additional options 
batch.Denoising.filter=[0.01, 0.1];                 % frequency filter (band-pass values, in Hz)
batch.Denoising.done=1;
batch.Denoising.overwrite='Yes';

% uncomment the following 3 lines if you prefer to run one step at a time:
% conn_batch(batch); % runs Denoising step only
% clear batch;
% batch.filename=fullfile(cwd,'Arithmetic_Scripted.mat');            % Existing conn_*.mat experiment name

% FIRST-LEVEL ANALYSIS step
% CONN Analysis                                     % Default options (uses all ROIs in conn/rois/ as connectivity sources); see conn_batch for additional options 
batch.Analysis.done=1;
batch.Analysis.overwrite='Yes';

% Run all analyses
conn_batch(batch);



