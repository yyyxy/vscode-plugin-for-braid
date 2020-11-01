class API:

    def __init__(self, package_name, class_name, class_description):
        self.package_name = package_name
        self.class_name = class_name
        self.class_description = class_description
        self.methods = []
        self.methods_descriptions_pure_text = []
        self.methods_descriptions = []  #the description is segmented into words
        self.methods_descriptions_stemmed = []
        self.methods_matrix = []
        self.methods_idf_vector = []
        self.class_description_matrix = None
        self.class_description_idf_vector = None


    def print_api(self):
        print(self.package_name+'.'+self.class_name,self.class_description)


class Question:

    def __init__(self,id,title,body,score,view_count,accepted_answer_id):
        self.id = id
        self.title = title
        self.body = body
        self.accepted_answer_id = accepted_answer_id
        self.score = score
        self.view_count = view_count
        self.answers = list()
        self.title_words = None
        self.matrix = None
        self.idf_vector = None

class Answer:

    def __init__(self, id, parent_id, body, score):
        self.id = id
        self.parent_id = parent_id
        self.body = body
        self.score = score




