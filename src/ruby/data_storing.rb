def save_build(build, builds_file, sep)
	build_colnames = ['id', 'travis_id', 'state', 'start_time', 'duration', 'commit_hash', 'config']
	if builds_file.pos == 0
		builds_file.write("#{build_colnames.join(sep)}\n")
	end
	start_time = build.started_at.nil? ? "" : build.started_at.strftime("%Y-%m-%d %H:%M")
	build_features = [build.number, build.id, build.state, start_time, build.duration, build.commit.sha, build.config.to_json]
	builds_file.write("#{build_features.join(sep)}\n")
end

def save_job(job, jobs_file, sep)
	job_colnames = ['id', 'travis_id', 'build_id', 'state', 'start_time', 'duration', 'commit_hash', 'config']
	if jobs_file.pos == 0
		jobs_file.write("#{job_colnames.join(sep)}\n")
	end
	start_time = job.started_at.nil? ? "" : job.started_at.strftime("%Y-%m-%d %H:%M")
	job_features = [job.number, job.id, job.build.number, job.state, start_time, job.duration, job.commit.sha, job.config.to_json]
	jobs_file.write("#{job_features.join(sep)}\n")
end

def save_test_execution(test_runs, job, build, exe_file, sep)
	exe_columns = ["test_name", "build", "job", "test_result", "duration"]
	if exe_file.pos == 0
		exe_file.write("#{exe_columns.join(sep)}\n")
	end
	unless test_runs.empty?
		exe_file.write(test_runs.map {|run|
			values = [run.test_name, build.number, job.number, run.test_result, run.test_duration]
			values.join(sep)
			}.join("\n") + "\n"
		)
	end
end