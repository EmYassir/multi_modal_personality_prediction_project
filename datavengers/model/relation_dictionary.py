import numpy as np


class RelationDictionary:

    def __init__(self, labels):
        # labels
        self._labels = np.array(labels)

        # all pages seen so far
        self._global_count = 0
        self._all_pages_counts = {}

        # page like counts given a label
        self._like_counts_per_label = {}
        for l in labels:
            self._like_counts_per_label[l] = {}

    # Update the dictionary
    def update(self, pages_string, label):
        page_list = pages_string.split(' ')
        for page_id in page_list:
            # Updating the global dictionary
            if page_id in self._all_pages_counts:
                self._all_pages_counts[page_id] += 1
            else:
                self._all_pages_counts[page_id] = 1
            # Update the global count
            self._global_count += 1

            # Updating the classes where the page belongs
            dic = self._like_counts_per_label[label]
            if page_id in dic:
                dic[page_id] += 1
            else:
                dic[page_id] = 1

    # Get number of times the page is liked per label
    def get_like_count_per_label(self, page_id, label):
        # Wrong class id
        if label not in self._labels:
            print('Unknown label: %s' % label)
            return -1
        dic = self._like_counts_per_label[label]
        if page_id in dic:
            return dic[page_id]
        else:
            print('Page %s not found under label: %s' % (page_id, label))
            return -1

    # Get total number of page likes per label
    def get_total_count_per_label(self, label):
        if label not in self._labels:
            print('Invalid label id given: could not find label id %d' % label)
            return 0
        return len(self._like_counts_per_label[label])

    # Total number of page likes
    def get_global_count(self):
        return self._global_count

    # Total number of pages
    def get_count_unique_pages(self):
        return len(self._all_pages_counts)

    # Returns list of n most popular pages per label (n <= 0 means no limit)
    def get_n_top_pages_per_label(self, label, n=0):
        res = []
        if label not in self._labels:
            print('Unknown label: %s' % label)
            return res

        dic = self._like_counts_per_label[label]
        res = sorted(dic, key=dic.get, reverse=True)
        if n <= 0:
            return res
        else:
            return res[:min(len(res), n)]

    # Returns list of n least popular pages per label (n <= 0 means no limit)
    def get_n_bottom_pages_per_label(self, label, n=0):
        res = []
        if label not in self._labels:
            print('Unknown label: %s' % label)
            return res
        dic = self._like_counts_per_label[label]
        res = sorted(dic, key=dic.get, reverse=False)
        if n <= 0:
            return res
        else:
            return res[:min(len(res), n)]

    # Returns list of n common pages between list of labels passed as parameter
    # Empty list means that all labels will be considered
    # n <= 0 means no limit
    def get_n_pages_common_to_labels(self, labels=[], n=0):
        res = []

        # Total complexity cost : Quadratic
        dic = self._all_pages_counts
        sorted_global_list = sorted(self._all_pages_counts, key=dic.get, reverse=True)

        # Label list to scan
        labels_to_scan = []
        if len(labels) == 0:
            labels_to_scan = self._labels
        else:
            for l in labels:
                if l not in self._labels:
                    continue
                else:
                    labels_to_scan.append(l)
        # updating limit to scan
        limit = n
        if limit <= 0:
            limit = self._global_count

        count = 0
        print('Scanning %d pages' % len(sorted_global_list))
        for word in sorted_global_list:
            is_word_common = True
            for l in labels_to_scan:
                if word not in self._like_counts_per_label[l]:
                    is_word_common = False
                    break
            # Adding word to the list
            if is_word_common:
                res.append(word)
                count += 1
            # Checking if max is reached
            if count >= limit:
                break
        # End
        return res

    # Returns list of n pages unique to given label
    # n <= 0 means no limit
    def get_n_pages_unique_to_label(self, label, n=0):
        res = []
        if label not in self._labels:
            print('Unknown label: %s' % label)
            return res
        # Getting the list of words belonging to the given label
        dic = self._like_counts_per_label[label]
        aux_list = sorted(dic, key=dic.get, reverse=True)
        for elem in aux_list:
            is_unique = True
            for la in self._labels:
                if la != label and elem in self._like_counts_per_label[la]:
                    is_unique = False
                    break
            if is_unique:
                res.append(elem)

        limit = len(res)
        if n > 0:
            limit = min(n, limit)
        return res[:limit]
