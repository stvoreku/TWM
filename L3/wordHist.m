function h = wordHist(feats, words)
    words_cnt = size(words, 1);
    dis = pdist2(feats, words, 'squaredeuclidean');
    [~, lbl] = min(dis, [], 2);
    h = histcounts(lbl, (1:words_cnt+1)-0.5, 'Normalization', 'probability');
end