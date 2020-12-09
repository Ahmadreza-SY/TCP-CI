require 'travis'
require 'fileutils'
require 'parallel'
require 'rover-df'
require 'tqdm'

TEST_LOG_REGEX = /Running (?<test_class>([A-Za-z]{1}[A-Za-z\d_]*\.)+[A-Za-z][A-Za-z\d_]*)(.*?)Tests run: (?<total_tests>\d*), Failures: (?<failed_tests>\d*), Errors: (?<error_tests>\d*), Skipped: (?<skipped_tests>\d*), Time elapsed: (?<tests_duration>[+-]?([0-9]*[.])?[0-9]+)/m
ANSI_COLOR_CODES_REGEX = /\x1B\[([0-9]{1,3}((;[0-9]{1,2})?){1,2})?[mGK]/
PROGRESS_LOGS_REGEX = /Progress \(\d*\):.*[\n\r]*/
SCRIPT_REGEX = /The command "(?<script>.*)" exited with \d*./
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

def fetch_logs_and_create_dataset(repository_slug, output_dir, concurrency=1)
    puts "starting to get #{repository_slug} repository CI build logs"
    result_path = "#{output_dir}/test_execution_history.csv"
    if File.file?(result_path)
        puts "skipping test execution history data extraction, #{result_path} already exists."
        return result_path
		end
		separator = ","
    repository = Travis::Repository.find(repository_slug)
    tests_file = File.open(result_path, "w")
    dataset_columns = ["name", "cycle", "verdict", "duration", "executionDateTime"]
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
                    progress_message.sub!(/\A.*\Z/, "Received #{received_megabytes.round(2)} MBs of logs")
                    unless test_runs.empty?
                        tests_file.write(test_runs.map {|run|
                            verdict = (run.failed_tests == 0 && run.error_tests == 0) ? 0 : 1
                            values = [run.test_class, build.number, verdict, (run.tests_duration * 1000).to_i, 
                                default_job.started_at.strftime("%Y-%m-%d %H:%M")]
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
    puts "finshed extracting #{repository_slug} test execution history data. Total logs received: #{received_megabytes.round(2)} MBs\n"
    return result_path
end

def download_logs_and_save_test_cases(repository_slug, output_dir)
    FileUtils.makedirs(output_dir)
    dataset_path = "#{output_dir}/test_execution_history.csv"
    if File.file?(dataset_path)
        puts "skipping #{repository_slug} repository, execution history #{dataset_path} already exists."
        return
    end
    fetch_logs_and_create_dataset(repository_slug, output_dir, 8)
end

if ARGV.length != 2
	puts "Exactly two arguments (project slug and output dir) is required!"
	exit
end

download_logs_and_save_test_cases(ARGV[0], ARGV[1])