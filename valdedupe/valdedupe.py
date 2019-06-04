#! /usr/bin/env python3
'''
Created on May 20, 2019

@author: rburr
'''
import csv
import json
from io import StringIO, open
import dedupe
import re
import click
import logging
import sys
import os
from flask import Flask
from flask.templating import render_template

app = Flask(__name__)

@app.route('/')
def valdedupe():
    return valDedupeWeb()


class ValDedupe(object):
    '''
    classdocs
    '''

    def __init__(self, input_file, skip_training, training_file, settings_file, sample_size, recall_weight, destructive, output_file, output_json):
        '''
        Constructor
        '''
        # Define fields to use to dedupe advanced.csv
        self.field_defs = [
             {'field' : 'first_name', 'type': 'String'},
             {'field' : 'last_name', 'type': 'String'},
             {'field' : 'company', 'type': 'String', 'has_missing': True},
             {'field' : 'email', 'type': 'String'},
             {'field' : 'address1', 'type': 'String', 'has_missing': True},
             {'field' : 'address2', 'type': 'String', 'has_missing': True},
             {'field' : 'zip', 'type': 'String', 'has missing':True},
             {'field' : 'city', 'type': 'String', 'has missing':True},
             {'field' : 'state_long', 'type': 'String', 'has missing':True},
             {'field' : 'state', 'type': 'String', 'has missing':True},
             {'field' : 'phone', 'type': 'String', 'has missing':True}
             ]
        
        self.field_names = [field_def['field'] for field_def in self.field_defs]
#         # List of fields in order to be used for json records for output to preserve ordering and to include fields not used for dedupe.
#         self.rec_fields= ['id','first_name','last_name','company','email','address1','address2','zip','city','state_long','state','phone']
       
        self.delimiter = ','
        self.skip_training = skip_training
        self.output_file = output_file
        self.sample_size = sample_size
        self.recall_weight = recall_weight
        self.destructive = destructive
        self.settings_file = settings_file
        self.output_json = output_json
        self.training_file = training_file

        try:
            self.input = open(input_file, encoding='utf-8-sig').read()
        except IOError:
            print(input_file)
            raise self.error("Could not open input", )



    def preProcess(self, column):
        
        # Data cleaning for casing, extra spaces, 
        # quotes, and new lines are ignored.
    
        column = re.sub('  +', ' ', column)
        column = re.sub('\n', ' ', column)
        column = column.strip().strip('"').strip("'").lower().strip()
        if column == '' :
            column = None
        return column
    

    def readRecords(self, input_file, delimiter=',', prefix=None):
        # Read in records from  CSV file and create a dictionary of records, 
        # where the key is a unique record ID and each value is a dict 
        # of the row fields.
    
        data = {}
        
        reader = csv.DictReader(StringIO(input_file), delimiter=delimiter)
        for i, row in enumerate(reader):
            clean_row = {k: self.preProcess(v) for (k, v) in row.items() if k is not None}
            if prefix:
                row_id = u"%s|%s" % (prefix, i)
            else:
                row_id = i
            data[row_id] = clean_row
    
        return data
    
    def exact_matches(self, data_d, match_fields):
        unique = {}
        redundant = {}
        for key, record in data_d.items():
            record_hash = hash(tuple(record[f] for f in match_fields))
            if record_hash not in redundant:
                unique[key] = record
                redundant[record_hash] = (key, [])
            else:
                redundant[record_hash][1].append(key)
    
        return unique, {k : v for k, v in redundant.values()}
    
    # If we have training data saved from a previous run of dedupe,
    # look for it an load it in.
    def dedupe_training(self, deduper) :

        # __Note:__ if you want to train from scratch, delete the training_file
        if os.path.exists(self.training_file):
            logging.info('reading labeled examples from %s' %
                         self.training_file)
            with open(self.training_file) as tf:
                deduper.readTraining(tf)

        if not self.skip_training:
            logging.info('starting active labeling...')

            dedupe.consoleLabel(deduper)

            # When finished, save our training away to disk
            logging.info('saving training data to %s' % self.training_file)
            if sys.version < '3' :
                with open(self.training_file, 'wb') as tf:
                    deduper.writeTraining(tf)
            else :
                with open(self.training_file, 'w') as tf:
                    deduper.writeTraining(tf)
        else:
            logging.info('skipping the training step')

        deduper.train()

        # After training settings have been established make a cache file for reuse
        logging.info('caching training result set to file %s' % self.settings_file)
        with open(self.settings_file, 'wb') as sf:
            deduper.writeSettings(sf)
    
    def displayDupes(self, dup_sets):
        print("Potential Duplicates\n")
        for dup_set in dup_sets:
            for dup in dup_set:
                dup_rec = recToStr(dup)
                print("\t- {}".format(dup_rec))
            print( '\n')
        
    def displayUniques(self, uniques):
        print("No Duplicates\n")
        for rec in uniques:
            unique_rec = recToStr(rec)
            print("\t- {}".format(unique_rec))
       
        
        

    def dedupeCsv(self):

        data_d = {}
        # import the specified CSV file

        data_d = self.readRecords(self.input, delimiter=self.delimiter)

        logging.info('imported %d rows', len(data_d))

        # sanity check for provided field names in CSV file
        for field in self.field_defs:
            if field['type'] != 'Interaction':
                if not field['field'] in data_d[0]:

                    raise self.error("Could not find field '" +
                                            field['field'] + "' in input")

        logging.info('using fields: %s' % [field['field']
                                           for field in self.field_defs])

        # If --skip_training has been selected, and we have a settings cache still
        # persisting from the last run, use it in this next run.
        # __Note:__ if you want to add more training data, don't use skip training
        if self.skip_training and os.path.exists(self.settings_file):

            # Load our deduper from the last training session cache.
            logging.info('reading from previous training cache %s'
                                                          % self.settings_file)
            with open(self.settings_file, 'rb') as f:
                deduper = dedupe.StaticDedupe(f)

            fields = {variable.field for variable in deduper.data_model.primary_fields}
            unique_d, parents = self.exact_matches(data_d, fields)
                
        else:
            # # Create a new deduper object and pass our data model to it.
            deduper = dedupe.Dedupe(self.field_defs)

            fields = {variable.field for variable in deduper.data_model.primary_fields}
            unique_d, parents = self.exact_matches(data_d, fields)

            # Set up our data sample
            logging.info('taking a sample of %d possible pairs', self.sample_size)
            deduper.sample(unique_d, self.sample_size)

            # Perform standard training procedures
            self.dedupe_training(deduper)

        # ## Blocking

        logging.info('blocking...')

        # ## Clustering

        # Find the threshold that will maximize a weighted average of our precision and recall. 
        # When we set the recall weight to 2, we are saying we care twice as much
        # about recall as we do precision.
        #
        # If we had more data, we would not pass in all the blocked data into
        # this function but a representative sample.

        logging.info('finding a good threshold with a recall_weight of %s' %
                     self.recall_weight)
        threshold = deduper.threshold(unique_d, recall_weight=self.recall_weight)

        # `duplicateClusters` will return sets of record IDs that dedupe
        # believes are all referring to the same entity.

        logging.info('clustering...')
        clustered_dupes = deduper.match(unique_d, threshold)

        expanded_clustered_dupes = []
        rows_used = []
        for cluster, scores in clustered_dupes:
            new_cluster = list(cluster)
            new_scores = list(scores)
            for row_id, score in zip(cluster, scores):
                children = parents.get(row_id, [])
                new_cluster.extend(children)
                new_scores.extend([score] * len(children))
                # Track parent rows processed
                rows_used.append(row_id)
            expanded_clustered_dupes.append((new_cluster, new_scores))
        # Add any parents with no clustered exact_dups but with exact dupes to expanded_clustered_dupes
        for row, exact_dups in parents.items():
            if row not in rows_used and exact_dups is not None and len(exact_dups) > 0:
                new_cluster = [row]
                new_cluster.extend(exact_dups)
                new_scores = [1.0]
                new_scores.extend([1.0] * len(exact_dups))
                expanded_clustered_dupes.append((new_cluster, new_scores))
                
        clustered_dupes = expanded_clustered_dupes

        logging.info('# duplicate sets %s' % len(clustered_dupes))

        write_function = writeResults
        # write out our results
        if self.destructive:
            write_function = writeUniqueResults
 
        if self.output_file:
            with open(self.output_file, 'w', encoding='utf-8') as output_file:
                write_function(clustered_dupes, self.input, output_file)
        else:
            write_function(clustered_dupes, self.input, sys.stdout)
            
        res = writeJsonResults( clustered_dupes, data_d, self.output_json)
        
        js_decoder = json.JSONDecoder()   
        results = js_decoder.decode(res)

        #with open(self.output_json,"r") as f:
        #    results = json.load(f)
        self.displayDupes( results['duplicates'])
        
        self.displayUniques( results['uniques'])
            
        
    def dedupeCsv2(self):

        # import the specified CSV file
        data_d = self.readRecords(self.input, delimiter=self.delimiter)
        # sanity check for provided field names in CSV file
        for field in self.field_defs:
            if field['type'] != 'Interaction':
                if not field['field'] in data_d[0]:

                    raise self.error("Could not find field '" +
                                            field['field'] + "' in input")

        # If --skip_training has been selected, and we have a settings cache still
        # persisting from the last run, use it in this next run.
        # __Note:__ if you want to add more training data, don't use skip training
        if os.path.exists(self.settings_file):
            with open(self.settings_file, 'rb') as f:
                deduper = dedupe.StaticDedupe(f)

            fields = {variable.field for variable in deduper.data_model.primary_fields}
            unique_d, parents = self.exact_matches(data_d, fields)

        # Find the threshold that will maximize a weighted average of our precision and recall. 
        # When we set the recall weight to 2, we are saying we care twice as much
        # about recall as we do precision.
        #
        # If we had more data, we would not pass in all the blocked data into
        # this function but a representative sample.

        threshold = deduper.threshold(unique_d, recall_weight=self.recall_weight)

        # `duplicateClusters` will return sets of record IDs that dedupe
        # believes are all referring to the same entity.

        clustered_dupes = deduper.match(unique_d, threshold)

        expanded_clustered_dupes = []
        rows_used = []
        for cluster, scores in clustered_dupes:
            new_cluster = list(cluster)
            new_scores = list(scores)
            for row_id, score in zip(cluster, scores):
                children = parents.get(row_id, [])
                new_cluster.extend(children)
                new_scores.extend([score] * len(children))
                rows_used.append(row_id)
            expanded_clustered_dupes.append((new_cluster, new_scores))
        # Add any parents with no clustered exact_dups but with exact dupes to expanded_clustered_dupes
        for row, exact_dups in parents.items():
            if row not in rows_used and exact_dups is not None and len(exact_dups) > 0:
                new_cluster = [row]
                new_cluster.extend(exact_dups)
                new_scores = [1.0]
                new_scores.extend([1.0] * len(exact_dups))
                expanded_clustered_dupes.append((new_cluster, new_scores))

        clustered_dupes = expanded_clustered_dupes
            
        res = writeJsonResults( clustered_dupes, data_d)
        
        js_decoder = json.JSONDecoder()   
        results = js_decoder.decode(res)

        return render_template('duplicates.html', dup_sets= results['duplicates'] , uniques= results['uniques'])
        

@app.cli.command("valdupe_train")
@click.argument('input_file', default='-', envvar='INPUT_FILE')
@click.option('--skip_training/--no_skip_training', default=False, help='Skip labeling examples by user and read training from training_files only')
@click.option('--training_file', default='training.json', help='Path to a new or existing file consisting of labeled training examples')
@click.option('--settings_file',default='learned_settings', help='Path to a new or existing file consisting of learned training settings')
@click.option('--sample_size', default=1500, help='Number of random sample pairs to train off of')
@click.option('--recall_weight', default=0.5, help='Threshold that will maximize a weighted average of our precision and recall')
@click.option('--destructive', default=False, help='Output file will contain unique records only')
@click.option('--output_file', type=click.types.STRING, default=None, help='CSV file to store deduplication results')
@click.option('--output_json', type=click.types.STRING, default=None, help='JSON file to store deduplication results')
def valDedupe(input_file, skip_training, training_file, settings_file, sample_size, recall_weight, destructive, output_file, output_json):
    """Identify potential duplicates and non-duplicate records in the specified CSV file argument. If omitted, will accept input on STDIN."""
    d = ValDedupe(input_file, skip_training, training_file, settings_file, sample_size, recall_weight, destructive, output_file, output_json)
    d.dedupeCsv()

def valDedupeWeb():
    """Identify potential duplicates and non-duplicate records in the specified CSV input_file and display as a web page"""
    input_file = 'data/advanced.csv' #/Users/rburr/eclipse-workspace/Validity/data/advanced.csv'
    skip_training= True
    training_file='training.json'
    settings_file= 'learned_settings'
    sample_size=1500
    recall_weight=0.5
    destructive= False
    output_file= None
    output_json= None
    d = ValDedupe(input_file, skip_training, training_file, settings_file, sample_size, recall_weight, destructive, output_file, output_json)
    return d.dedupeCsv2()

# ## Writing results
def writeResults(clustered_dupes, input_file, output_file):

    # Write our original data back out to a CSV with a new column called 
    # 'Cluster ID' which indicates which records refer to each other.

    logging.info('saving results to: %s' % output_file)

    cluster_membership = {}
    for cluster_id, (cluster, score) in enumerate(clustered_dupes):
        for record_id in cluster:
            cluster_membership[record_id] = cluster_id

    unique_record_id = cluster_id + 1

    writer = csv.writer(output_file)

    reader = csv.reader(StringIO(input_file))

    heading_row = next(reader)
    heading_row.insert(0, u'Cluster ID')
    writer.writerow(heading_row)

    for row_id, row in enumerate(reader):
        if row_id in cluster_membership:
            cluster_id = cluster_membership[row_id]
        else:
            cluster_id = unique_record_id
            unique_record_id += 1
        row.insert(0, cluster_id)
        writer.writerow(row)


# ## Writing results
def writeUniqueResults(clustered_dupes, input_file, output_file):

    # Write our original data back out to a CSV with a new column called 
    # 'Cluster ID' which indicates which records refer to each other.

    logging.info('saving unique results to: %s' % output_file)

    cluster_membership = {}
    for cluster_id, (cluster, score) in enumerate(clustered_dupes):
        for record_id in cluster:
            cluster_membership[record_id] = cluster_id

    unique_record_id = cluster_id + 1

    writer = csv.writer(output_file)

    reader = csv.reader(StringIO(input_file))

    heading_row = next(reader)
    heading_row.insert(0, u'Cluster ID')
    writer.writerow(heading_row)

    seen_clusters = set()
    for row_id, row in enumerate(reader):
        if row_id in cluster_membership:
            cluster_id = cluster_membership[row_id]
            if cluster_id not in seen_clusters:
                row.insert(0, cluster_id)
                writer.writerow(row)
                seen_clusters.add(cluster_id)
        else:
            cluster_id = unique_record_id
            unique_record_id += 1
            row.insert(0, cluster_id)
            writer.writerow(row)

# ## Writing results
def writeJsonResults(clustered_dupes, data, output_json_file=None):

    # Write our original data back out to a CSV with a new column called 
    # 'Cluster ID' which indicates which records refer to each other.

    logging.info('saving results to: %s' % output_json_file)

    duplicate_sets = []
    uniques = []
    seen_rec = set()
    for (cluster, score) in clustered_dupes:
        dup_set = []
        for record_id in cluster:
            dup_set.append( data[ record_id])
            seen_rec.add(record_id)
        duplicate_sets.append(dup_set)
    for row_id, row in enumerate(data):
        if row_id not in seen_rec:       
            uniques.append(data[row])
    
    # Save results as json
    results = {}
    results['duplicates'] = duplicate_sets
    results['uniques'] = uniques
    
    if output_json_file is not None:
        with open(output_json_file,"w") as f:
            json.dump(results,f)
    
    js_encoder = json.JSONEncoder()   
    return js_encoder.encode(results)

@app.template_filter('recToStr')
def recToStr(rec):
    field_list = ['id','first_name','last_name','company','email','address1','address2','zip','city','state_long','state','phone']
    rec_str = ''
    first = True
    for field in field_list:
        rec_str += ', ' if not first else ''
        first = False
        rec_str += rec[field] or ''
    return rec_str


if __name__ == "__main__":
    valDedupe()

