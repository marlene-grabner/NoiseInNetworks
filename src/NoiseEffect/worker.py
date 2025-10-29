import traceback
import NoiseEffect.individual_run_setup as ia


def workerFunction(request, random_seed_list):
    try:
        noise_information = request.pop("noise_information")
        network_request = request

        analysis_obj = ia.IndividualAnalysis(
            network_request=network_request,
            noise_information=noise_information,
            random_seed_list=random_seed_list,
        )
        analysis_obj.run()
        return (analysis_obj.identifier, analysis_obj.results)
    except Exception as e:
        # Return error information instead of crashing
        identifier = (
            f"ERROR_{request.get('type', 'unknown')}_{request.get('instance', 'X')}"
        )
        error_info = {
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
            "request": request,
        }
        print(f"ERROR in worker: {identifier} - {e}")
        return (identifier, error_info)
