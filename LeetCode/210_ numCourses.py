'''
现在你总共有 n 门课需要选，记为 0 到 n-1。
在选修某些课程之前需要一些先修课程。 例如，想要学习课程 0 ，你需要先完成课程 1 ，我们用一个匹配来表示他们: [0,1]
给定课程总量以及它们的先决条件，返回你为了学完所有课程所安排的学习顺序。
可能会有多个正确的顺序，你只要返回一种就可以了。如果不可能完成所有课程，返回一个空数组。

示例 1:
输入: 2, [[1,0]] 
输出: [0,1]
解释: 总共有 2 门课程。要学习课程 1，你需要先完成课程 0。因此，正确的课程顺序为 [0,1] 。
'''

# Still has the question
class Solution(object):
    def findOrder(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        """
        res = []
        def recurse(res, numCourses_new):
            find_course = numCourses_new
            if find_course <= 0:
                return res
            for banch_classes in prerequisites:
                if find_course in banch_classes:
                    #print("> . <", banch_classes.index(find_course))
                    target_course_idx = banch_classes.index(find_course)
                    if target_course_idx < len(banch_classes)-1:
                        res += [res_course for res_course in banch_classes[
                            target_course_idx+1:len(banch_classes)
                        ]]
                        #print(">> . <<", res)
                        res = recurse(res, res[-1])
            return res

        if len(prerequisites)==0:
            res += [c for c in range(numCourses)]
        else:
            for new_class in range(numCourses):
                res += [new_class]
                res = recurse(res, new_class)
            #print(res)
        out = []

        for i in range(len(res)):
            temp = res.pop()
            if temp not in out:
                out += [temp]
                #print(out)
        
        if len(out) == numCourses:
            return out
        else:
            return []
