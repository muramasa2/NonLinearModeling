close all;
clear;

%% 音源の読み込み

music = 'Take_five'

origin_filename = strcat(music, '.wav');
filename1 = strcat(music, '_teac_curve.wav');
filename2 = strcat(music, '_teac_maudio.wav');
filename3 = strcat('inv_', music, '_nuforce_curve.wav');
filename4 = strcat('inv_', music, '_nuforce_maudio.wav');

[origin_signal, Fs] = audioread(origin_filename);
[signal_1, Fs] = audioread(filename1); 
[signal_2, Fs] = audioread(filename2);
[signal_3, Fs] = audioread(filename3); 
[signal_4, Fs] = audioread(filename4);

origin_signal = origin_signal(:,1);

norm_origin_signal = origin_signal - mean(origin_signal);
norm_signal_1 = signal_1 - mean(signal_1);
norm_signal_2 = signal_2 - mean(signal_2);
norm_signal_3 = signal_3 - mean(signal_3);
norm_signal_4 = signal_4 - mean(signal_4);

x_start = 543000;
x_end = 546000;

norm_origin_signal = norm_origin_signal(714000:724000)
norm_signal_1 = norm_signal_1(714000:724000)
norm_signal_2 = norm_signal_2(714000:724000)
norm_signal_3 = norm_signal_3(714000:724000)
norm_signal_4 = norm_signal_4(714000:724000)


% 振幅正規化・相互相関の確認・音源の差分作成
t1 = finddelay(norm_origin_signal, norm_signal_1);
t2 = finddelay(norm_origin_signal, norm_signal_2);
t3 = finddelay(norm_origin_signal, norm_signal_3);
t4 = finddelay(norm_origin_signal, norm_signal_4);

disp(t1)
disp(t2)
disp(t3)
disp(t4)

base = min([t1, t2, t3, t4])

figure(1)
subplot(5, 1, 1);
plot(origin_signal);
% xlim([19500, 22000])
% xlim([543000, 546000])
xlim([714000, 724000])
subplot(5, 1, 2);
plot(signal_1);
% xlim([19500, 22000])
% xlim([543000, 546000])
xlim([714000, 724000])
subplot(5, 1, 3);
plot(signal_2);
% xlim([19500, 22000])
% xlim([543000, 546000])
xlim([714000, 724000])
subplot(5, 1, 4);
plot(signal_3);
% xlim([19500, 22000])
% xlim([543000, 546000])
xlim([714000, 724000])
subplot(5, 1, 5);
plot(signal_4);
% xlim([19500, 22000])
% xlim([543000, 546000])
xlim([712000, 724000])

fix_origin_signal = origin_signal(1-base:end);
i = 1
fix_t = zeros(1,4)

for t = [t1, t2, t3, t4]
    if not(t==base)
        fix_t(i) = base-t
    end
    i = i+1
end

fix_signal_1 = signal_1(1-fix_t(1):end);
fix_signal_2 = signal_2(1-fix_t(2):end);
fix_signal_3 = signal_3(1-fix_t(3):end);
fix_signal_4 = signal_4(1-fix_t(4):end);

figure(2)
subplot(5, 1, 1);
plot(fix_origin_signal);
% xlim([543000, 546000])
% xlim([19500, 22000])
xlim([714000, 724000])
subplot(5, 1, 2);
plot(fix_signal_1);
% xlim([543000, 546000])
% xlim([19500, 22000])
xlim([714000, 724000])
subplot(5, 1, 3);
plot(fix_signal_2);
% xlim([543000, 546000])
% xlim([19500, 22000])
xlim([714000, 724000])
subplot(5, 1, 4);
plot(fix_signal_3);
% xlim([543000, 546000])
% xlim([19500, 22000])
xlim([714000, 724000])
subplot(5, 1, 5);
plot(fix_signal_4);
% xlim([543000, 546000])
% xlim([19500, 22000])
xlim([714000, 724000])

Fs = 44100;
out_origin_filename = strcat('fix_', music, '.wav');
out_filename1 = strcat('fix_', music, '_teac_curve.wav');
out_filename2 = strcat('fix_', music, '_teac_maudio.wav');
out_filename3 = strcat('fix_inv_', music, '_nuforce_curve.wav');
out_filename4 = strcat('fix_inv_', music, '_nuforce_maudio.wav');

audiowrite(out_origin_filename, fix_origin_signal, Fs);
audiowrite(out_filename1, fix_signal_1, Fs);
audiowrite(out_filename2, fix_signal_2, Fs);
audiowrite(out_filename3, fix_signal_3, Fs);
audiowrite(out_filename4, fix_signal_4, Fs);