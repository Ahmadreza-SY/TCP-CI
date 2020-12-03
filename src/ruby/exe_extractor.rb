require 'travis'
require 'fileutils'
require 'parallel'
require 'rover-df'
require 'tqdm'

BASE_DIR = "data"
TEST_LOG_REGEX = /Running (?<test_class>([A-Za-z]{1}[A-Za-z\d_]*\.)+[A-Za-z][A-Za-z\d_]*)(.*?)Tests run: (?<total_tests>\d*), Failures: (?<failed_tests>\d*), Errors: (?<error_tests>\d*), Skipped: (?<skipped_tests>\d*), Time elapsed: (?<tests_duration>[+-]?([0-9]*[.])?[0-9]+)/m
ANSI_COLOR_CODES_REGEX = /\x1B\[([0-9]{1,3}((;[0-9]{1,2})?){1,2})?[mGK]/
PROGRESS_LOGS_REGEX = /Progress \(\d*\):.*[\n\r]*/
SCRIPT_REGEX = /The command "(?<script>.*)" exited with \d*./
CSV_SEPARATOR = ","
TestRun = Struct.new(:total_tests, :failed_tests, :error_tests, :skipped_tests, :tests_duration, :test_class)

def extract_all_tests(job)
    log = job.log.body
    if log.nil?
        return [], 0
    end
    cleaned_log = log.gsub(ANSI_COLOR_CODES_REGEX, '').gsub(PROGRESS_LOGS_REGEX, '')
    received_bytes = log.bytesize
    script = if job.config.key?("script") 
        job.config["script"].first  
    else 
        matched = cleaned_log.match(SCRIPT_REGEX)
        if matched.nil? 
            return [], received_bytes
        end
        matched[:script]
    end
    script_log = cleaned_log.match(/travis_time:start:(\w+)[\n\r]\$ #{script}[\n\r](?<log>.*?)travis_time:end/m)
    if script_log.nil? 
        return [], received_bytes
		end
		matches = script_log["log"].scan(TEST_LOG_REGEX)
		return matches.map { |match|
			TestRun.new(
					match[1].to_i,
					match[2].to_i,
					match[3].to_i,
					match[4].to_i,
					match[5].to_f,
					match[0]
			)
    }.compact, received_bytes
end

def fetch_logs_and_create_dataset(repository_name, repo_dir, separator, concurrency=1)
    puts "starting to get #{repository_name} repository"
    result_path = "#{repo_dir}/test_execution_history.csv"
    if File.file?(result_path)
        puts "skipping download phase, #{result_path} already exists."
        return result_path
    end
    repository = Travis::Repository.find(repository_name)
    tests_file = File.open(result_path, "w")
    dataset_columns = ["name", "duration", "executionDateTime", "verdict", "cycle"]
    tests_file.write("#{dataset_columns.join(separator)}\n")
    received_megabytes = 0.0
    progress_message = "Initailizing ..."
    Parallel.each(repository.each_build, in_threads: concurrency, progress: progress_message) { |build|
        begin
            unless build.nil?
                default_job = build.jobs.first
                if !default_job.nil?
                    test_runs, received_bytes = extract_all_tests(default_job)
                    received_megabytes += received_bytes.to_f / 2**20
                    progress_message.sub!(/\A.*\Z/, "Received #{received_megabytes.round(2)} MBs")
                    unless test_runs.empty?
                        tests_file.write(test_runs.map {|run|
                            verdict = (run.failed_tests == 0 && run.error_tests == 0) ? 0 : 1
                            values = [run.test_class, (run.tests_duration * 1000).to_i, 
                                default_job.started_at.strftime("%Y-%m-%d %H:%M"), verdict, build.number]
                            values.join(separator)
                        	}.join("\n") + "\n"
                        )
                    end
                end
            end
        rescue StandardError => e
            sleep_seconds = 15
            puts "Exception occurred: #{e.message}, retrying in #{sleep_seconds} seconds ..."
            sleep(sleep_seconds)
            retry
        end
    }
    tests_file.close
    puts "finshed getting #{repository_name} repository. Total received size: #{received_megabytes.round(2)} MBs\n"
    return result_path
end

def download_logs_and_save_test_cases(repository_name, separator=CSV_SEPARATOR)
    repo_dir = "#{BASE_DIR}/#{repository_name}"
    FileUtils.makedirs(repo_dir)
    dataset_path = "#{repo_dir}/test_case_dataset.csv"
    if File.file?(dataset_path)
        puts "skipping #{repository_name} repository, dataset #{dataset_path} already exists."
        return
    end
    fetch_logs_and_create_dataset(repository_name, repo_dir, separator, 8)
end

repositories_file = File.open("repositories.txt")
repositories = repositories_file.readlines.map(&:chomp)
repositories_file.close
FileUtils.makedirs(BASE_DIR)
repositories.each do |repo|
    download_logs_and_save_test_cases(repo)
end